constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void BlurRow(read_only image2d_t image, write_only image2d_t output, const unsigned int kernel_width, __constant float *filter, uint image_width)
{	
	const int x = get_global_id(0);
	const int y = get_global_id(1);

	const uint radius = (kernel_width - 1) / 2;
	const int groupX = get_local_size(0) * get_group_id(0);
	const uint offset = groupX - radius;
	if (get_global_id(0) < image_width) {
		float4 res = (float4)0;

		int i = 0;
		for (; i + 7 < kernel_width; i += 8) {
			for (int j = 0; j < 8; ++j) {
				res += filter[i + j] * read_imagef(image, sampler, (int2)(i + offset + j + get_local_id(0), y));
			}
		}

		for (; i < kernel_width; ++i) {
			res += filter[i] * read_imagef(image, sampler, (int2)(i + offset + get_local_id(0), y));;
		}

		write_imagef(output, (int2)(x,y), res);
	}
}

__kernel void BlurColumn(read_only image2d_t image, write_only image2d_t output, const unsigned int kernel_width, __constant float *filter, uint image_height, uint image_width) {
	const int x = get_global_id(0);
	const int y = get_global_id(1);

	const uint radius = (kernel_width - 1) / 2;
	const int groupX = get_local_size(0)*get_group_id(0);
	const int groupY = get_local_size(1)*get_group_id(1);

	const uint offset = groupY - radius;

	if (get_global_id(1) < image_height) {
		float4 res = (float4) 0;

		int i = 0;

		for (; i + 7 < kernel_width; i += 8) {
			for (int j = 0; j < 8; ++j) {
				res += filter[i + j] * read_imagef(image, sampler, (int2)(x, offset + i + j + get_local_id(1)));
			}
		}

		for (; i < kernel_width; ++i) {
			res += filter[i] * read_imagef(image, sampler, (int2)(x, offset + i + get_local_id(1)));
		}

		write_imagef(output, (int2)(x, y), res + (float4)(0,0,0,0.5f));
	}
}