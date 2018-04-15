__kernel void test(__global float * a) {
	float4 c = as_float4(intel_sub_group_block_read4((__global uint *)a +
		get_sub_group_id() * get_sub_group_size())) ;
	float4 b = intel_sub_group_shuffle(c, get_sub_group_local_id());
	printf("%d.%d %f %f %f %f\n", get_sub_group_id(), get_sub_group_local_id(), b.x, b.y, b.z, b.w);
}