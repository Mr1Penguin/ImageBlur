constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void BlurRow(__global float4 * image, __global float4 * output, size_t kernel_width,
	__constant float *filter, uint image_width, uint image_align_width, __local float4 * temp)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);

	const uint radius = (kernel_width - 1) / 2;
	const int groupX = get_local_size(0) * get_group_id(0);
	const int wsize = get_local_size(0);
	const unsigned int loadSize = wsize + kernel_width;

	for (int i = get_local_id(0); i < loadSize; i = i + get_local_size(0))
	{
		int cx = clamp(i + groupX - radius, (uint)0, image_width - 1);
		temp[i] = image[y*image_width + cx];
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if (get_global_id(0) < image_width) {
		float4 res = (float4)0;

		int i = 0;
		for (; i + 7 < kernel_width; i += 8) {
			for (int j = 0; j < 8; ++j) {
				res += filter[i + j] * temp[i + j + get_local_id(0)];
			}
		}

		for (; i < kernel_width; ++i) {
			res += filter[i] * temp[i + get_local_id(0)];
		}

		output[y*image_align_width + x] = res;
	}
}

__kernel void BlurColumn(read_only image2d_t image, write_only image2d_t output, size_t kernel_width, __constant float *filter, uint image_height, uint image_width) {
	const int x = get_global_id(0);
	const int y = get_global_id(1);

	const uint radius = (kernel_width - 1) / 2;
	const int groupY = get_local_size(1)*get_group_id(1);

	const uint offset = groupY - radius + get_local_id(1);

	if (get_global_id(1) < image_height && get_global_id(0) < image_width) {
		float4 res = (float4) 0;

		int i = 0;

		for (; i + 7 < kernel_width; i += 8) {
			for (int j = 0; j < 8; ++j) {
				res += filter[i + j] * read_imagef(image, sampler, (int2)(x, offset + i + j));
			}
		}

		for (; i < kernel_width; ++i) {
			res += filter[i] * read_imagef(image, sampler, (int2)(x, offset + i));
		}

		write_imagef(output, (int2)(x, y), res + (float4)(0,0,0,0.5f));
	}
}