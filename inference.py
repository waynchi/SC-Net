"""# Inference"""

# testing model predict with seaborn and plots
model = keras.models.load_model(model_location)
import seaborn as  sb
import matplotlib.pyplot as plt
print(image_shape)
def generate_noise():
    input_image = np.full(image_shape, 0)
    # input_image = np.expand_dims(images[0], -1)

    # Random noise
    # input_image = np.random.rand(28, 28)
    # input_image[input_image >= 0.5] = 1
    # input_image[input_image < 0.5] = 0

    input_image = input_image.astype(np.uint8)
    input_image = np.expand_dims(input_image, 0)
    return input_image

test_image = generate_noise()
if is_grayscale:
    softmax_predictions, sigmoid_predictions = model.predict(test_image)
    softmax_predictions = softmax_predictions.reshape(image_shape[0], image_shape[1])
    heatmap = sb.heatmap(softmax_predictions)
    plt.show()
    softmax_predictions = np.exp(softmax_predictions)
    softmax_predictions = softmax_predictions / np.sum(softmax_predictions)
    print(sigmoid_predictions.shape)
    # print(sigmoid_predictions[])
    sigmoid_predictions = np.argmax(sigmoid_predictions, axis=-1)
    print(sigmoid_predictions.shape)
    sigmoid_predictions = sigmoid_predictions.reshape(image_shape[0], image_shape[1])
    heatmap = sb.heatmap(softmax_predictions)
    plt.show()
    heatmap = sb.heatmap(sigmoid_predictions)
    plt.show()
else:
    softmax_predictions = model.predict(test_image)
    softmax_predictions = softmax_predictions.reshape(28, 28)
    softmax_predictions = np.exp(softmax_predictions)
    softmax_predictions = softmax_predictions / np.sum(softmax_predictions)
    heatmap = sb.heatmap(softmax_predictions)
    plt.show()

import os
import shutil
import time


def inference(model, input_image, directory, iterations, temp_start=2, temp_end=0.5, top_k=250, is_grayscale=True, is_debug=False):
    create_image(input_image, "{}/input.png".format(directory), image_shape=image_shape[:-1])

    # temperatures = np.linspace(temp_start, temp_end, num=iterations)
    temperatures = np.geomspace(temp_start, temp_end, num=iterations)
    temperatures_reverse = (temp_start + temp_end) - temperatures[::-1]
    temperatures = np.concatenate((temperatures_reverse[:int(temperatures.shape[0]/2)], temperatures[int(temperatures.shape[0]/2):]))
    
    working_image = deepcopy(input_image)
    working_images = []
    num_added = 0
    num_removed = 0
    for i in range(iterations):
        temp = temperatures[i]            
        binary_image = deepcopy(working_image)
        binary_image[binary_image > 0] = 1
        if is_grayscale:
            softmax_predictions, sigmoid_predictions = model.predict(working_image)
        else:
            softmax_predictions = model.predict(binary_image)

        softmax_predictions = softmax_predictions.flatten()
        if is_grayscale:
            # sigmoid_predictions = np.argmax(sigmoid_predictions, axis=-1)
            sigmoid_predictions = sigmoid_predictions.reshape(-1, 256)

        softmax_predictions = np.exp(softmax_predictions / temp)
        softmax_predictions = softmax_predictions / np.sum(softmax_predictions)
        # softmax_predictions = np.nan_to_num(softmax_predictions)
        # print(softmax_predictions)
        # print(softmax_predictions[-1])
        indices = np.arange(softmax_predictions.shape[0])

        zipped = zip(softmax_predictions, indices)
        zipped = list(reversed(sorted(zipped, key = lambda x : x[0])))
        zipped = zipped[:top_k]
        zipped = sorted(zipped, key=lambda x : x[1])
        softmax_predictions, indices = zip(*zipped)
        softmax_predictions = np.asarray(softmax_predictions)
        softmax_predictions = softmax_predictions / np.sum(softmax_predictions)
        indices = np.asarray(indices)

        index = np.random.choice(indices, p=softmax_predictions)
        working_image = working_image.flatten()
        # if index == softmax_predictions.shape[0]:
        #     print("stopping")
            # break
        if is_grayscale:
            if working_image[index] != 0:
                num_removed += 1
            elif working_image[index] == 0:
                num_added += 1
            sigmoid_probs = sigmoid_predictions[index]
            sigmoid_indices = np.arange(sigmoid_probs.shape[0])
            working_image[index] = np.random.choice(sigmoid_indices, p=sigmoid_probs)
        else:
            if working_image[index] == 1:
                num_removed += 1
                working_image[index] = 0
            elif working_image[index] == 0:
                num_added += 1
                working_image[index] = 1
            else:
                print(working_image[index])
        working_image = np.reshape(working_image, [1, *image_shape])
        if i % 50 == 0:
            if is_debug:
                print("softmax")
                softmax_predictions = softmax_predictions.reshape(image_shape[:-1])
                heatmap = sb.heatmap(deepcopy(softmax_predictions))
                plt.show()
                print("sigmoid")
                sigmoid_predictions = np.argmax(sigmoid_predictions, axis=-1).reshape(image_shape[:-1])
                heatmap = sb.heatmap(deepcopy(sigmoid_predictions))
                plt.show()
            create_image(working_image, os.path.join(directory, "working_{}.png".format(i)), image_shape=image_shape[:-1])

    final_image = working_image
    final_binary_image = deepcopy(final_image)
    final_binary_image[final_binary_image > 0] = 1
    create_image(final_binary_image, os.path.join(directory, "final_binary.png"), image_shape=image_shape[:-1])

    print(final_image.shape)
    print("num added: {}. num removed: {}".format(num_added, num_removed))
    img = create_image(final_image, os.path.join(directory, 'final.png'), image_shape=image_shape[:-1])
    return img, deepcopy(final_image)

# model = keras.models.load_model(model_location)

drive_folder = '/content/drive/My Drive'

model_names = [# 'sc-model-es-net-60000-4',
               #'sc-model-es-net-60000-16',
               # 'sc-model-es-net-mnist-grayscale-double-softmax',
               #'sc-model-nade-60000-4',
               'checkpoints/nade-cifar-grayscale-double-softmax-0-single'
               # 'checkpoints/sc-model-es-net-cifar-grayscale-double-softmax-1'
               ]

config = {
    'sc-model-es-net-60000-4': {
        "iterations": 300,
        "temp_start": 0.99,
        "temp_end": 0.99,
        "top_k": 10000
    },
    'checkpoints/nade-cifar-grayscale-double-softmax-0-single': {
        "iterations": 1500,
        "temp_start": 0.99,
        "temp_end": 0.99,
        "top_k": 10000
    },
    'checkpoints/sc-model-es-net-cifar-grayscale-double-softmax-1': {
        "iterations": 1500,
        "temp_start": 2,
        "temp_end": 1,
        "top_k": 10000
    },
    'sc-model-nade-60000-4': {
        "iterations": 170,
        "temp_start": 0.99,
        "temp_end": 0.99,
        "top_k" : 10000
    },

}

sample_sqrt = 5
for model_name in model_names:
    model = keras.models.load_model(os.path.join(drive_folder, model_name + '.hdf5'))
    model_config = config[model_name]
    generated_images = []
    for i in range(sample_sqrt**2):
        directory = "images_{}".format(i)
        os.makedirs(directory, exist_ok=True)
        input_image = generate_noise()
        # input_image = np.expand_dims(np.expand_dims(images[i], 0), -1)

        img, _ = inference(model, input_image, directory, 
                           model_config['iterations'], temp_start=model_config['temp_start'], 
                           temp_end=model_config['temp_end'], top_k=model_config['top_k'], 
                           is_grayscale=is_grayscale, is_debug=False)
        generated_images.append(img)
    
    final_im = Image.new('RGB', (image_shape[0] * sample_sqrt, image_shape[1] * sample_sqrt))

    y_offset = 0
    for i in range(sample_sqrt):
        x_offset = 0
        new_im = Image.new('RGB', (image_shape[0] * sample_sqrt, image_shape[1]))
        for j in range(sample_sqrt):
            im = deepcopy(generated_images[(i * sample_sqrt) + j])
            new_im.paste(im, (x_offset, 0))
            x_offset += image_shape[0]
        final_im.paste(new_im, (0, y_offset))
        y_offset += image_shape[0]
        
    model_name = model_name.split('/')[-1]
    final_im.save(model_name + '.png')
