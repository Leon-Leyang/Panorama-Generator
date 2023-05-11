import cv2
import numpy as np


# Class to generate a panorama from warped frames
class PanoramaGenerator:
    @staticmethod
    def gen_panorama(warped_frames, num_levels=3):
        panorama = warped_frames[0]
        for i in range(1, len(warped_frames)):
            panorama = PanoramaGenerator.__blend_frames_multi_band(panorama, warped_frames[i], num_levels)

        return panorama

    @staticmethod
    # Function to generate a Gaussian pyramid
    def __gen_gaussian_pyramid(frame, num_levels):
        pyramid = [frame]
        for _ in range(num_levels - 1):
            frame = cv2.pyrDown(frame)
            pyramid.append(frame)

        return pyramid

    @staticmethod
    # Function to generate a Laplacian pyramid
    def __gen_laplacian_pyramid(frame, num_levels):
        frame = frame.astype(np.float32)
        gaussian_pyramid = PanoramaGenerator.__gen_gaussian_pyramid(frame, num_levels)
        laplacian_pyramid = []
        for i in range(num_levels - 1):
            expanded_frame = cv2.pyrUp(gaussian_pyramid[i + 1])
            # Resize the expanded frame to the size of the current level
            expanded_frame = cv2.resize(expanded_frame, (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0]))
            laplacian_pyramid.append(gaussian_pyramid[i] - expanded_frame)

        laplacian_pyramid.append(gaussian_pyramid[num_levels - 1])

        return laplacian_pyramid

    @staticmethod
    # Function to blend two laplacian pyramids
    def __blend_laplacian_pyramids(pyramid_1, pyramid_2, pyramid_mask):
        blended_pyramid = []
        for i in range(len(pyramid_1)):
            blended_pyramid.append(pyramid_1[i] * (1 - pyramid_mask[i]) + pyramid_2[i] * pyramid_mask[i])

        return blended_pyramid

    @staticmethod
    # Function to reconstruct a frame from a laplacian pyramid
    def __reconstruct_from_laplacian_pyramid(laplacian_pyramid):
        frame = laplacian_pyramid[-1]
        for i in range(len(laplacian_pyramid) - 2, -1, -1):
            expanded_frame = cv2.pyrUp(frame)
            # Resize the expanded frame to the size of the current level
            expanded_frame = cv2.resize(expanded_frame, (laplacian_pyramid[i].shape[1], laplacian_pyramid[i].shape[0]))
            frame = laplacian_pyramid[i] + expanded_frame

        return frame.astype(np.uint8)

    @staticmethod
    # Function to generate a smooth mask of the same size as the frame
    def __gen_smooth_mask(frame1, frame2):
        # Generate a mask and initialize it to 0
        # Then set the non-overlapping area of frame1 to 0 and the non-overlapping area of frame2 to 1
        mask = np.zeros_like(frame1, dtype=np.float32)
        mask[(frame1 > 0) & (frame2 == 0)] = 0
        mask[(frame1 == 0) & (frame2 > 0)] = 1

        # Apply a Gaussian blur to the mask
        ksize = 55
        mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)

        # Normalize the mask to be in the range [0, 1]
        mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))

        return mask

    # Function to generate a feathered mask of the same size as the frame
    @staticmethod
    def __gen_feathered_mask(frame1, frame2):
        # Convert frames to grayscale
        frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Create a binary mask with non-overlapping areas set to 1 and 0
        mask = np.zeros_like(frame1, dtype=np.float32)
        mask[(frame1_gray > 0) & (frame2_gray == 0)] = 0
        mask[(frame1_gray == 0) & (frame2_gray > 0)] = 1

        # Create a linear gradient in the overlapping area
        overlap_mask = np.logical_and(frame1_gray > 0, frame2_gray > 0)
        rows, cols = np.where(overlap_mask)

        if cols != []:
            min_col, max_col = np.min(cols), np.max(cols)

            for col in range(min_col, max_col + 1):
                mask[overlap_mask[:, col], col, :] = (col - min_col) / (max_col - min_col)

        return mask

    @staticmethod
    # Function to get the mask of the valid region of a frame
    def __get_valid_mask(frame):
        vld_mask = np.any(frame != 0, axis=-1)

        # Stack the mask three times to get a 3-channel mask
        vld_mask = np.stack((vld_mask, vld_mask, vld_mask), axis=-1).astype(np.uint8)
        return vld_mask

    @staticmethod
    # Function to blend two frames using multi-band blending
    def __blend_frames_multi_band(frame_1, frame_2, num_levels):
        mask = PanoramaGenerator.__gen_feathered_mask(frame_1, frame_2)

        # Generate the Gaussian and Laplacian pyramids for the two frames
        laplacian_pyramid_1 = PanoramaGenerator.__gen_laplacian_pyramid(frame_1, num_levels)
        laplacian_pyramid_2 = PanoramaGenerator.__gen_laplacian_pyramid(frame_2, num_levels)
        mask_pyramid = PanoramaGenerator.__gen_gaussian_pyramid(mask, num_levels)

        # Blend the two laplacian pyramids
        blended_laplacian_pyramid = PanoramaGenerator.__blend_laplacian_pyramids(laplacian_pyramid_1,
                                                                                 laplacian_pyramid_2,
                                                                                 mask_pyramid)

        # Reconstruct the blended frame from the blended laplacian pyramid
        blended_frame = PanoramaGenerator.__reconstruct_from_laplacian_pyramid(blended_laplacian_pyramid)

        return blended_frame
