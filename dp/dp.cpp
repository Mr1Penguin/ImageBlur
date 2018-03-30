// dp.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iomanip>

template<class F, class ...N>
cl::size_t<sizeof...(N) + 1> GetSizeT(F first, N... args) {
	std::vector<F> tmp{ first, static_cast<F>(args)... };
	int i = 0;
	cl::size_t<sizeof...(N) + 1> ret;
	for (auto && x : tmp)
		ret[i++] = x;
	return ret;
}

#define MagickEpsilon  (1.0e-15)
#define MagickSQ2PI  2.50662827463100024161235523934010416269302368164062
#define QuantumRange  65535.0f
#define QuantumScale  ((double) 1.0/(double) QuantumRange)

size_t GetOptimalWidth(float radius, float sigma) {
	if (radius > MagickEpsilon) {
		return size_t(2.0*ceil(radius) + 1);
	}

	auto gamma = std::abs(sigma);
	if (gamma <= MagickEpsilon) {
		return size_t{ 3 };
	}

	auto perceptible_reciprocal = [](double arg) { return std::abs(arg) >= MagickEpsilon ? 1.0 / arg : std::copysign(1.0, arg) / MagickEpsilon;  };
	auto alpha = perceptible_reciprocal(2 * gamma*gamma);
	auto beta = perceptible_reciprocal(MagickSQ2PI*gamma);
	size_t width = 5;

	while (true) {
		double normalize = 0.0;
		int j = (width - 1) / 2;
		for (int i = -j; i <= j; ++i) {
			normalize += std::exp(-double(i*i)*alpha)*beta;
		}
		auto val = std::exp(-double(j*j)*alpha)*beta / normalize;
		if (val < QuantumScale || val < MagickEpsilon) {
			break;
		}
		width += 2;
	}

	return size_t{ width - 2 };
}

std::vector<float> GenerateKernel(float radius, float sigma) {
	auto sigma_ = std::abs(sigma);
	size_t width;
	if (radius >= 1.0f) {
		width = radius * 2 + 1;
	}
	else {
		width = GetOptimalWidth(radius, sigma);
	}

	std::vector<float> kernel;
	kernel.resize(width);
	auto x = (width - 1) / 2;
	auto y = 0;

	int length = (width * 3/*rank*/ - 1) / 2;
	if (sigma > MagickEpsilon) {
		sigma *= 3;
		auto alpha = 1.0 / (2.0*sigma*sigma);
		auto beta = 1.0 / (MagickSQ2PI*sigma);
		for (int u = -length; u <= length; u++) {
			kernel[(u + length) / 3] += std::exp(-double(u*u)*alpha)*beta;
		}
	}
	else {
		kernel[x + y * width] = 1.0;
	}

	//scaling
	auto min = 0.0, max = 0.0;
	auto pos_range = 0.0, neg_range = 0.0;
	for (auto i = 0; i < width; ++i) {
		if (std::abs(kernel[i]) < MagickEpsilon) {
			kernel[i] = 0.0;
		}
		kernel[i] < 0 ? (neg_range += kernel[i]) : (pos_range += kernel[i]);
		min = *std::min_element(std::begin(kernel), std::end(kernel));
		max = *std::max_element(std::begin(kernel), std::end(kernel));
	}

	auto pos_scale = std::abs(pos_range) >= MagickEpsilon ? pos_range : 1.0;
	auto neg_scale = std::abs(neg_range) >= MagickEpsilon ? -neg_range : 1.0;

	pos_scale = 1.0 / pos_scale;
	neg_scale = 1.0 / neg_scale;

	for (int i = 0; i < width; ++i) {
		if (!std::isnan(kernel[i])) {
			kernel[i] *= kernel[i] >= 0 ? pos_scale : neg_scale;
		}
	}

	return kernel;
}

int main(int argc, char *argv[])
{
	if (argc < 6) return 1;

	std::vector<cl::Platform> all_platforms;
	cl::Platform::get(&all_platforms);
	if (all_platforms.size() == 0) {
		std::cout << " No platforms found. Check OpenCL installation!\n";
		exit(1);
	}

	int i{ 1 };

	for (auto & pl : all_platforms) {
		std::cout << i++ << ": " << pl.getInfo<CL_PLATFORM_NAME>() << std::endl;
		std::vector<cl::Device> all_devices;
		pl.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
		int j{ 1 };
		for (auto & d : all_devices) {
			std::cout << "  " << j++ << ": " << d.getInfo<CL_DEVICE_NAME>() << "; " << d.getInfo<CL_DEVICE_OPENCL_C_VERSION>() << std::endl;
		}

	}

	std::string str;
	int x, y;

	std::cout << "select device(x.y): ";
	std::cin >> str;
	x = stoi(str.substr(0, 1));
	y = stoi(str.substr(2, 1));

	cl::Platform default_platform = all_platforms[x - 1];
	std::cout << "Using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";

	std::vector<cl::Device> all_devices;
	default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
	if (all_devices.size() == 0) {
		std::cout << " No devices found. Check OpenCL installation!\n";
		exit(1);
	}
	cl::Device default_device = all_devices[y - 1];
	std::cout << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << "\n";

#ifndef MAGICK
	std::ifstream stream("main.cl");
#else
	std::ifstream stream("d:\\kernels.cl");
#endif
	std::string code((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());


	cl::Context context({ default_device });
	cl::Program::Sources sources;
	sources.push_back({ code.c_str(), code.length() });

	cl::Program program(context, sources);
#ifdef MAGICK
	std::string opts = "-cl-single-precision-constant -cl-mad-enable -DMAGICKCORE_HDRI_SUPPORT=1 -DCLQuantum=float -DCLSignedQuantum=float -DCLPixelType=float4 -DQuantumRange=65535.000000f -DQuantumScale=0.000015 -DCharQuantumScale=1.000000 -DMagickEpsilon=0.000000 -DMagickPI=3.141593 -DMaxMap=65535 -DMAGICKCORE_QUANTUM_DEPTH=16";
	if (program.build({ default_device }, opts.c_str()) != CL_SUCCESS) {
#else
	if (program.build({ default_device }) != CL_SUCCESS) {
#endif
		std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << "\n";
		exit(1);
	}

	FILE *fp; fopen_s(&fp, argv[1], "rb");
	char header[8];
	fread(header, 1, 8, fp);
	auto png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
	auto info_ptr = png_create_info_struct(png_ptr);
	setjmp(png_jmpbuf(png_ptr));
	png_init_io(png_ptr, fp);
	png_set_sig_bytes(png_ptr, 8);
	png_read_info(png_ptr, info_ptr);
	auto width = png_get_image_width(png_ptr, info_ptr);
	auto height = png_get_image_height(png_ptr, info_ptr);
	png_read_update_info(png_ptr, info_ptr);
	setjmp(png_jmpbuf(png_ptr));
	auto raw_width = png_get_rowbytes(png_ptr, info_ptr);
	auto rows = new png_bytep[height];
	for (int i = 0; i < height; ++i) rows[i] = new png_byte[raw_width];
	png_read_image(png_ptr, rows);
	std::vector<float> pixels(raw_width*height);
	for (unsigned i = 0; i < height; ++i) {
		for (unsigned j = 0; j < raw_width; ++j) {
			pixels[i*raw_width + j] = rows[i][j];
		}
		//delete[] rows[i];
	}
	//delete[] rows;
	fclose(fp);

	auto radius = std::stof(argv[2], nullptr);
	auto sigma = std::stof(argv[3], nullptr);
	auto local_size = std::stol(argv[4], nullptr, 10);

	auto kernel = GenerateKernel(radius, sigma);
	std::cout << "kernel(" << kernel.size() << "):";
	for (auto & k : kernel) std::cout << k << ";";
	std::cout << std::endl;
	cl_int res = 0;

	cl_uint pitch_alignment;

	res = clGetDeviceInfo(default_device(), CL_DEVICE_IMAGE_PITCH_ALIGNMENT, sizeof(cl_uint), &pitch_alignment, nullptr);
	//res = default_device.getInfo(CL_DEVICE_IMAGE_PITCH_ALIGNMENT, &pitch_alignment);
	
	cl::ImageFormat format;
	format.image_channel_data_type = CL_FLOAT;
	format.image_channel_order = CL_RGBA;
	auto align_width = pitch_alignment * (1 + ((width - 1) / pitch_alignment));
	std::vector<float> tmp(width * 4 * height);
	cl::Buffer input(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, width * height * sizeof(cl_float4), nullptr, &res);  //(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, format, width, height);
	//cl::Buffer temp_buffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, pitch_alignment * (1 + ((width - 1) / pitch_alignment)) * height * sizeof(float) * 4, nullptr, &res);
#ifndef MAGICK
	auto temp_buffer = clCreateBuffer(context(), CL_MEM_READ_WRITE, align_width * height * sizeof(float) * 4, nullptr, &res);
	cl::Image2D output(context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, format, width, height, 0, nullptr, &res);
#else
	auto temp_buffer = clCreateBuffer(context(), CL_MEM_READ_WRITE, width * height * sizeof(float) * 4, nullptr, &res);
	cl::Buffer out_buffer (context, CL_MEM_WRITE_ONLY, width * height * sizeof(float) * 4, nullptr, &res);
#endif
	//cl::Image2D temp_image(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, format, width, height);
	cl::Buffer kern(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, kernel.size() * sizeof(float), nullptr, &res);

	
	cl::CommandQueue queue(context, default_device);
	res = queue.enqueueWriteBuffer(input, CL_TRUE, 0, width*height * sizeof(cl_float4), pixels.data());
	//auto res = queue.enqueueWriteImage(input, CL_TRUE, GetSizeT(0,0,0), GetSizeT(width, height, 1), 0/*sizeof(float) * 4 * width*/, 0, pixels.data());
	res = queue.enqueueWriteBuffer(kern, CL_TRUE, 0, kernel.size() * sizeof(float), kernel.data());
	cl::Kernel blur_row(program, "BlurRow");
#ifndef MAGICK
	res = blur_row.setArg(0, input);
	res = blur_row.setArg(1, temp_buffer);
	res = blur_row.setArg(2, kernel.size());
	res = blur_row.setArg(3, kern);
	res = blur_row.setArg(4, width);
	res = blur_row.setArg(5, align_width);
	res = blur_row.setArg(6, sizeof(cl_float4) * (kernel.size() + local_size), nullptr);
#else
	res = blur_row.setArg(0, input);
	res = blur_row.setArg(1, 4U);
	res = blur_row.setArg(2, 0x7ffffff);
	res = blur_row.setArg(3, kern);
	res = blur_row.setArg(4, static_cast<unsigned>(kernel.size()));
	res = blur_row.setArg(5, width);
	res = blur_row.setArg(6, height);
	res = blur_row.setArg(7, sizeof(cl_float4) * (kernel.size() + local_size), nullptr);
	res = blur_row.setArg(8, temp_buffer);
#endif
	//for (local_size = 1; local_size <= 256; local_size *= 2) {
		auto glob = cl::NDRange(local_size*((width + local_size - 1) / local_size), height);
		//for (int i = 0; i < 10; ++i) {
			//std::cout << "Row[" << std::setfill('0') << std::setw(3) << local_size << ", " << std::setfill('0') << std::setw(3) << i + 1 << "]...";
			res = queue.enqueueNDRangeKernel(blur_row, cl::NullRange, glob, cl::NDRange(local_size, 1));
			if (res) std::cout << "BlurRow1 problem\n";
			res = queue.finish();
			if (res) std::cout << "BlurRow2 problem\n";
			std::cout << "done\n";
		//}
	//}
	//clEnqueueReadBuffer(queue(), temp_buffer, CL_TRUE, 0, )

#ifndef MAGICK
	auto exts = default_device.getInfo<CL_DEVICE_EXTENSIONS>();
	const char* ext_substr = strstr(&exts[0], "cl_khr_image2d_from_buffer");
	const char* ext_substr_end = ext_substr + strlen("cl_khr_image2d_from_buffer");
	if (!(ext_substr && (ext_substr_end[0] == ' ' || ext_substr_end[0] == 0)))
	{// check that the device supports the image from buffer extension
		throw std::runtime_error("The device does not support cl_khr_image2d_from_buffer extension");
		return 0;
	}

	cl_image_desc desc{ 0 };
	desc.image_type = CL_MEM_OBJECT_IMAGE2D;
	desc.image_width = width;
	desc.image_height = height;
	desc.image_row_pitch = align_width * sizeof(cl_float4);
	desc.buffer = temp_buffer;
	cl::Image2D::cl_type temp_image = clCreateImage(context(), CL_MEM_READ_WRITE, &format, &desc, nullptr, &res);
	size_t orig[3]{ 0,0,0 }, reg[3]{ width, height, 1 };
	//clEnqueueMapImage(queue(), temp_image, CL_TRUE, orig, reg, 0, align_width * sizeof(cl_float4))
	//res = clEnqueueReadBuffer(queue(), temp_buffer, CL_TRUE, 0, 
	/*for (int u = 0; u < 4 * width*height; u += 4) {
		std::cout << "[" << (u / 4) % width << "," << (u / 4) / width << "]: " << tmp[u] << "," << tmp[u + 1] << "," << tmp[u + 2] << "," << tmp[u + 3] << "\n";
	}*/
#endif
	cl::Kernel blur_col(program, "BlurColumn");
#ifndef MAGICK
	res = blur_col.setArg(0, temp_image);
	res = blur_col.setArg(1, output);
	res = blur_col.setArg(2, kernel.size());
	res = blur_col.setArg(3, kern);
	res = blur_col.setArg(4, height);
	res = blur_col.setArg(5, width);
#else
	res = blur_col.setArg(0, temp_buffer);
	res = blur_col.setArg(1, 4U);
	res = blur_col.setArg(2, 0x7ffffff);
	res = blur_col.setArg(3, kern);
	res = blur_col.setArg(4, static_cast<unsigned>(kernel.size()));
	res = blur_col.setArg(5, width);
	res = blur_col.setArg(6, height);
	res = blur_col.setArg(7, sizeof(cl_float4) * (kernel.size() + local_size), nullptr);
	res = blur_col.setArg(8, out_buffer);
#endif
	int lx = 1, ly = 1;
	for (; lx <= 256; lx *= 2) {
		for (ly = 1; lx * ly <= 256; ly *= 2) {
			glob = cl::NDRange(lx*((width + lx - 1) / lx), ly*((height + ly - 1) / ly));
			for (int i = 0; i < 1; ++i) {
				std::cout << "Col[" << std::setfill('0') << std::setw(3) << lx << ", " << std::setfill('0') << std::setw(3) << ly << ", " << std::setfill('0') << std::setw(3) << i + 1 << "]...";
				res = queue.enqueueNDRangeKernel(blur_col, cl::NullRange, glob, cl::NDRange(lx, ly));
				if (res) std::cout << "BlurCol1 problem\n";
				res = queue.finish();
				if (res) std::cout << "BlurCol2 problem\n";
				res = queue.enqueueReadImage(output, CL_TRUE, GetSizeT(0, 0, 0), GetSizeT(width, height, 1), sizeof(float) * 4 * width, 0, tmp.data());
				fopen_s(&fp, (std::string(argv[5]) + std::to_string(lx) + "x" + std::to_string(ly) + ".png").c_str(), "wb");
				png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
				png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
				info_ptr = png_create_info_struct(png_ptr);
				setjmp(png_jmpbuf(png_ptr));
				png_init_io(png_ptr, fp);
				png_set_IHDR(png_ptr, info_ptr, width, height,
					8, PNG_COLOR_TYPE_RGBA, PNG_INTERLACE_NONE,
					PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);
				png_write_info(png_ptr, info_ptr);
				for (unsigned i = 0; i < height; ++i) {
					for (unsigned j = 0; j < raw_width; ++j) {
						rows[i][j] = tmp[i*(tmp.size() / height) + j];
						//std::cout << "[" << i << "][" << j << "]:" << (unsigned)rows[i][j] << std::endl;
					}
					png_write_row(png_ptr, rows[i]);
				}
				png_write_end(png_ptr, info_ptr);
				fclose(fp);
				std::cout << "done\n";
			}
		}
	}
	/*//for (local_size = 1; local_size <= 256; local_size *= 2) {
		//auto glob = cl::NDRange(width, local_size*((height + local_size - 1) / local_size));
		auto glob = cl::NDRange(local_size*((width + local_size - 1) / local_size), local_size*((height + local_size - 1) / local_size));
		//for (int i = 0; i < 100; ++i) {
			//std::cout << "Col[" << std::setfill('0') << std::setw(3) << local_size << ", " << std::setfill('0') << std::setw(3) << i + 1 << "]...";
			res = queue.enqueueNDRangeKernel(blur_col, cl::NullRange, glob, cl::NDRange(local_size, local_size));
			if (res) std::cout << "BlurCol1 problem\n";
			res = queue.finish();
			if (res) std::cout << "BlurCol2 problem\n";
			std::cout << "done\n";
		//}
	//}*/
#ifndef MAGICK
	res = queue.enqueueReadImage(output, CL_TRUE, GetSizeT(0, 0, 0), GetSizeT(width, height, 1), sizeof(float) * 4 * width, 0, tmp.data());
#else
	res = queue.enqueueReadBuffer(out_buffer, CL_TRUE, 0, width*height * sizeof(cl_float4), tmp.data());
#endif
	//res = clEnqueueReadImage(queue(), temp_image, CL_TRUE, orig, reg, align_width * sizeof(cl_float4), 0, tmp.data(), 0, nullptr, nullptr);
	/*for (int u = 0; u < 4 * width*height; u+=4) {
		std::cout << tmp[u] << "," << tmp[u+1] << "," << tmp[u+2] << "," << tmp[u+3] << "\n";
	}*/

	fopen_s(&fp, argv[5], "wb");
	png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
	png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
	info_ptr = png_create_info_struct(png_ptr);
	setjmp(png_jmpbuf(png_ptr));
	png_init_io(png_ptr, fp);
	png_set_IHDR(png_ptr, info_ptr, width, height,
		8, PNG_COLOR_TYPE_RGBA, PNG_INTERLACE_NONE,
		PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);
	png_write_info(png_ptr, info_ptr);
	for (unsigned i = 0; i < height; ++i) {
		for (unsigned j = 0; j < raw_width; ++j) {
			rows[i][j] = tmp[i*(tmp.size() / height) + j];
			//std::cout << "[" << i << "][" << j << "]:" << (unsigned)rows[i][j] << std::endl;
		}
		png_write_row(png_ptr, rows[i]);
		delete[] rows[i];
	}
	png_write_end(png_ptr, info_ptr);
	delete[] rows;
	fclose(fp);

	/*ilGenImages(1, &ImgId);
	ilBindImage(ImgId);
	ilTexImage(width, height, 1, 4, IL_RGBA, IL_FLOAT, tmp.data());
	ilEnable(IL_FILE_OVERWRITE);
	ilSave(IL_PNG, argv[5]);*/

    return 0;
}

