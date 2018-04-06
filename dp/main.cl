inline int ClampToCanvas(const int offset, const int range)
{
	return clamp(offset, (int)0, range - 1);
}

inline CLQuantum ClampToQuantum(const float value) { return (CLQuantum)(clamp(value, 0.0f, QuantumRange) + 0.5f); }

inline unsigned int getPixelIndex(const unsigned int number_channels, const unsigned int columns, const unsigned int x, const unsigned int y) { return (x * number_channels) + (y * columns * number_channels); }

inline float getPixelRed(const __global CLQuantum *p) { return (float)*p; }
inline float getPixelGreen(const __global CLQuantum *p) { return (float)*(p + 1); }
inline float getPixelBlue(const __global CLQuantum *p) { return (float)*(p + 2); }
inline float getPixelAlpha(const __global CLQuantum *p, const unsigned int number_channels) { return (float)*(p + number_channels - 1); }
inline void setPixelRed(__global CLQuantum *p, const CLQuantum value) { *p = value; }
inline void setPixelGreen(__global CLQuantum *p, const CLQuantum value) { *(p + 1) = value; }
inline void setPixelBlue(__global CLQuantum *p, const CLQuantum value) { *(p + 2) = value; }
inline void setPixelAlpha(__global CLQuantum *p, const unsigned int number_channels, const CLQuantum value) { *(p + number_channels - 1) = value; }

typedef enum {
	UndefinedChannel = 0x0000,
	RedChannel = 0x0001,
	GrayChannel = 0x0001,
	CyanChannel = 0x0001,
	GreenChannel = 0x0002,
	MagentaChannel = 0x0002,
	BlueChannel = 0x0004,
	YellowChannel = 0x0004,
	BlackChannel = 0x0008,
	AlphaChannel = 0x0010,
	OpacityChannel = 0x0010,
	IndexChannel = 0x0020,
	ReadMaskChannel = 0x0040,
	WriteMaskChannel = 0x0080,
	MetaChannel = 0x0100,
	CompositeChannels = 0x001F,
	AllChannels = 0x7ffffff,
	TrueAlphaChannel = 0x0100,
	RGBChannels = 0x0200,
	GrayChannels = 0x0400,
	SyncChannels = 0x20000,
	DefaultChannels = AllChannels
} ChannelType;

inline void WriteChannels(__global CLQuantum *p, const unsigned int number_channels, const ChannelType channel, float red, float green, float blue, float alpha)
{
	if ((channel & RedChannel) != 0)
		setPixelRed(p, ClampToQuantum(red));
	if (number_channels > 2)
	{
		if ((channel & GreenChannel) != 0)
			setPixelGreen(p, ClampToQuantum(green));
		if ((channel & BlueChannel) != 0)
			setPixelBlue(p, ClampToQuantum(blue));
	}
	if (((number_channels == 4) || (number_channels == 2)) && ((channel & AlphaChannel) != 0))
		setPixelAlpha(p, number_channels, ClampToQuantum(alpha));
}

inline void WriteFloat4(__global CLQuantum *image, const unsigned int number_channels, const unsigned int columns, const unsigned int x, const unsigned int y, const ChannelType channel, float4 pixel)
{
	__global CLQuantum *p = image + getPixelIndex(number_channels, columns, x, y);
	WriteChannels(p, number_channels, channel, pixel.x, pixel.y, pixel.z, pixel.w);
}

inline void ReadChannels(const __global CLQuantum *p, const unsigned int number_channels, const ChannelType channel, float *red, float *green, float *blue, float *alpha)
{
	if ((channel & RedChannel) != 0)
		*red = getPixelRed(p);
	if (number_channels > 2)
	{
		if ((channel & GreenChannel) != 0)
			*green = getPixelGreen(p);
		if ((channel & BlueChannel) != 0)
			*blue = getPixelBlue(p);
	}
	if (((number_channels == 4) || (number_channels == 2)) && ((channel & AlphaChannel) != 0))
		*alpha = getPixelAlpha(p, number_channels);
}

inline float4 ReadFloat4(const __global CLQuantum *image, const unsigned int number_channels, const unsigned int columns, const unsigned int x, const unsigned int y, const ChannelType channel)
{
	const __global CLQuantum *p = image + getPixelIndex(number_channels, columns, x, y);
	float red = 0.0f;
	float green = 0.0f;
	float blue = 0.0f;
	float alpha = 0.0f;
	ReadChannels(p, number_channels, channel, &red, &green, &blue, &alpha);
	return (float4)(red, green, blue, alpha);
}

__kernel void BlurRow(const __global CLQuantum *image, const unsigned int number_channels, const ChannelType channel, 
	__constant float *filter, const unsigned int width, const unsigned int imageColumns, const unsigned int imageRows, __global float4 *tempImage)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	const int columns = imageColumns;
	const unsigned int radius = (width - 1) / 2;
	const int groupX = get_local_size(0) * get_group_id(0);

	const uint offset = get_local_id(0) + groupX - radius;

	if (get_global_id(0) < columns)
	{
		float4 result = (float4)0;
		int i = 0;
		for (; i + 7 < width;)
		{
			for (int j = 0; j < 8; j++)
			{
				int cx = ClampToCanvas(i + j + offset, columns);
				result += filter[i + j] * ReadFloat4(image, number_channels, columns, cx, y, channel);
			}
			i += 8;
		}
		for (; i < width; i++)
		{
			int cx = ClampToCanvas(i + offset, columns);
			result += filter[i] * ReadFloat4(image, number_channels, columns, cx, y, channel);
		}
		tempImage[y * columns + x] = result;
	}
}
__kernel void BlurColumn(const __global float4 *blurRowData, const unsigned int number_channels, const ChannelType channel, 
	__constant float *filter, const unsigned int width, const unsigned int imageColumns, const unsigned int imageRows, __global CLQuantum *filteredImage)
{
	const int x = get_global_id(0);
	const int y = get_global_id(1);
	const int columns = imageColumns;
	const int rows = imageRows;
	unsigned int radius = (width - 1) / 2;
	const int groupX = get_local_size(0) * get_group_id(0);
	const int groupY = get_local_size(1) * get_group_id(1);
	if (get_global_id(1) < rows)
	{
		float4 result = (float4)0;
		int i = 0;
		for (; i + 7 < width;)
		{
			for (int j = 0; j < 8; j++)
				result += filter[i + j] * blurRowData[ClampToCanvas(i + j + get_local_id(1) + groupY - radius, rows) * columns + groupX];
			i += 8;
		}
		for (; i < width; i++)
			result += filter[i] * blurRowData[ClampToCanvas(i + get_local_id(1) + groupY - radius, rows) * columns + groupX];
		WriteFloat4(filteredImage, number_channels, columns, x, y, channel, result);
	}
}