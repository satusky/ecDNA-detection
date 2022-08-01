#!/usr/bin/env python3
import os
import glob
import argparse
import math
import csv
from PIL import Image, ImageEnhance
import numpy as np
from skimage import filters, measure, morphology, segmentation, transform
import matplotlib.pyplot as plt


def find_images(image_directory):
    """ Find images in parent and child directories """
    return glob.glob("{0}/**/*.tif".format(image_directory), recursive=True)

def check_unique_path(path):
    """ Adds a number suffix if the file already exists. Prevents overwrites. """
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = "{0}({1}){2}".format(filename, str(counter), extension)
        counter += 1

    return path

def mask_chromosomes_and_nuclei(image_array, min_object_size, max_object_size):
    """ Creates binary masks for chromosomes and intact nuclei """
    # Multi Otsu threshold image (2 thresholds)
    tss = filters.threshold_multiotsu(image_array)
    thresholded_image = np.digitize(image_array, bins=tss)
    thresholded_image[thresholded_image == 1] = 0
    thresholded_image[thresholded_image > 0] = 1

    label_objects = measure.label(thresholded_image)
    sizes = np.bincount(label_objects.ravel())

    # Create masks
    big_enough = sizes > min_object_size
    small_enough = sizes < max_object_size
    background = sizes == max(sizes)

    chromosome_sizes = np.logical_and(big_enough, small_enough)
    nuclei_sizes = np.logical_and(~small_enough, ~background)

    nuclei_mask = np.asarray(nuclei_sizes[label_objects])
    chromosome_mask = np.asarray(chromosome_sizes[label_objects])
    combo_mask = nuclei_mask + chromosome_mask

    # Dilate masks
    dilated_nuclei_mask = morphology.binary_dilation(nuclei_mask, morphology.square(30))
    dilated_chromosome_mask = morphology.binary_dilation(chromosome_mask, morphology.square(15))
    dilated_combo_mask = dilated_nuclei_mask + dilated_chromosome_mask

    return combo_mask, dilated_combo_mask

def white_tophat_filter(image_array, combined_mask, dilated_mask, footprint_size=5, block_size=None, sd_thresh=3):
    """ Removes large connected regions through a white tophat filter """
    if not block_size:
        block_size = math.gcd(image_array.shape[0], image_array.shape[1])

    footprint = morphology.disk(footprint_size)
    tophat_image = morphology.white_tophat(image_array, footprint)
    tophat_image[combined_mask] = np.mean(tophat_image)

    block_med = measure.block_reduce(tophat_image, block_size=(block_size, block_size), func=np.median)
    block_med = transform.resize(block_med, (image_array.shape[0], image_array.shape[1]))

    block_sd = measure.block_reduce(tophat_image, block_size=(block_size, block_size), func=np.std)
    block_sd = transform.resize(block_sd, (image_array.shape[0], image_array.shape[1]))

    above_threshold = np.ones_like(tophat_image)
    above_threshold[tophat_image < block_med + (sd_thresh * block_sd)] = 0
    above_threshold = segmentation.clear_border(above_threshold)
    above_threshold[dilated_mask] = 0

    return above_threshold

def clean_final_objects(above_threshold, min_spot_size, max_spot_size):
    """ Remove ecDNA spots outside the minimum/maximum allowed sizes """
    final_map = measure.label(above_threshold)
    props = measure.regionprops(final_map)

    for prop in props:
        if prop.area <= min_spot_size or prop.area >= max_spot_size:
            final_map[final_map == prop.label] = 0
        else:
            final_map[final_map == prop.label] = 1

    return final_map

def save_annotated_images(output_path, image_array, centroids, width=20):
    """ Saves PNG and SVG version of the image with found ecDNA circled """
    svg_output = check_unique_path(output_path + ".svg")
    png_output = check_unique_path(output_path + ".png")

    x = [centroid[1] for centroid in centroids]
    y = [centroid[0] for centroid in centroids]
    ratio = image_array.shape[0]/image_array.shape[1]
    _, ax = plt.subplots(figsize=(width * ratio, width))
    ax.imshow(image_array)
    ax.scatter(x, y, s=80, edgecolors='r', facecolors='none')
    ax.set_xlim(0, image_array.shape[1])
    ax.set_ylim(image_array.shape[0], 0)
    plt.ioff()
    plt.savefig(svg_output)
    plt.savefig(png_output)

def write_csv(data_dict, output_file, output_directory):
    """ Write file name and found ecDNA counts as a CSV """
    csv_path = check_unique_path(os.path.join(output_directory, output_file))
    with open(csv_path, 'w') as csvfile:
        csv_columns = ["file_name", "ecdna_count"]
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in data_dict:
            writer.writerow({"file_name": data, "ecdna_count": data_dict[data]})


def main(args):
    """ Main function """
    files = find_images(args.image_directory)
    ecdna_found_dict = {}
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    for file in files:
        print(f"Working on file {file}")
        img = Image.open(file)
        sharp_img = ImageEnhance.Contrast(img).enhance(2.0)
        sharp_array = np.array(sharp_img)[..., 0]

        # Mask chromosomes and intact nuclei
        combo_mask, dil_combo_mask = mask_chromosomes_and_nuclei(
            image_array=sharp_array,
            min_object_size=args.lower_chromosome_size,
            max_object_size=args.upper_chromosome_size
            )

        # White tophat filter to remove large objects
        above_threshold = white_tophat_filter(
            image_array=sharp_array,
            combined_mask=combo_mask,
            dilated_mask=dil_combo_mask,
            footprint_size=5,
            block_size=None,
            sd_thresh=args.sd_threshold
            )

        # Find connected regions above threshold
        final_map = clean_final_objects(
            above_threshold,
            min_spot_size=args.lower_ecdna_size,
            max_spot_size=args.upper_ecdna_size
            )

        final_spots = measure.label(final_map)
        props = measure.regionprops(final_spots)
        centroid_list = [prop.centroid for prop in props]

        image_output_name = "_".join(file.replace(".tif", "").split("/")[-2:])
        image_output_path = os.path.join(args.output_directory, image_output_name)
        save_annotated_images(
            output_path=image_output_path,
            image_array=sharp_array,
            centroids=centroid_list,
            width=20
            )

        ecdna_found_dict[file] = len(centroid_list)

    csv_file_name = "ecdna_found.csv"
    write_csv(
        data_dict=ecdna_found_dict,
        output_file=csv_file_name,
        output_directory=args.output_directory
        )


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument(
        "--image_directory",
        "-d",
        help="Top directory to search for images",
        type=str,
        required=True
        )
    PARSER.add_argument(
        "--output_directory",
        "-o",
        help="Directory to save output files",
        type=str,
        default="./output"
        )
    PARSER.add_argument(
        "--lower_chromosome_size",
        "-cl",
        help="Minimum pixel area for chromosome masking",
        type=int,
        default=75
        )
    PARSER.add_argument(
        "--upper_chromosome_size",
        "-cu",
        help="Maximum pixel area for chromosome masking (anything above is considered nucleus)",
        type=int,
        default=20000
        )
    PARSER.add_argument(
        "--lower_ecdna_size",
        "-el",
        help="Minimum pixel area for ecDNA masking",
        type=int,
        default=10
        )
    PARSER.add_argument(
        "--upper_ecdna_size",
        "-eu",
        help="Maximum pixel area for ecDNA masking",
        type=int,
        default=300
        )
    PARSER.add_argument(
        "--sd_threshold",
        "-s",
        help="Number of standard deviations above median to threshold ecDNA spots",
        type=int,
        default=3
        )
    ARGS = PARSER.parse_args()

    main(ARGS)
