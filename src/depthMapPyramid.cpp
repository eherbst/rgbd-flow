/*
 * depthMapPyramid: downsample depth maps without uncertainty information but accounting for invalid measurements
 *
 * Evan Herbst
 * 12 / 19 / 12
 */

#include "rgbd_util/mathUtils.h"
#include "rgbd_util/primesensorUtils.h"
#include "celiu_flow/depthMapPyramid.h"

/*
 * values <= 0 will be considered invalid
 */
void DepthMapPyramid::ConstructPyramid(const DImage &image, double ratio, int minWidth)
{
	ASSERT_ALWAYS(image.nchannels() == 1);

	// the ratio cannot be arbitrary numbers
	if(ratio>0.98 || ratio<0.4)
		ratio=0.75;
	// first decide how many levels
	double w = image.width() * ratio;
	int nLevels = 1;
	while(w >= minWidth)
	{
		nLevels++;
		w *= ratio;
	}

	ImPyramid.resize(nLevels);
	ImPyramid[0].copyData(image);
	for(int m=1;m<nLevels;m++)
	{
		const DImage& largerImg = ImPyramid[m - 1];
		const int largerWidth = ImPyramid[m - 1].width(), largerHeight = ImPyramid[m - 1].height();

		const double levelRatio = pow(ratio, m);
		DImage foo(image.width() * levelRatio, image.height() * levelRatio, image.nchannels());

		const float factor = (float)largerHeight / foo.height();
		for(int32_t i = 0; i < foo.height(); i++)
			for(int32_t j = 0; j < foo.width(); j++)
			{
				//discretize the exact location onto a pixel's square; if it's invalid, return invalid; else avg nearby valid values
				const int32_t x = std::max(0, std::min(largerWidth - 1, (int32_t)rint(factor * j))), y = std::max(0, std::min(largerHeight - 1, (int32_t)rint(factor * i)));
				if(largerImg(y, x) > 0)
				{
					//local sampling: get four pixels
					const int32_t xs[2] = {std::max(0, std::min(largerWidth - 1, (int32_t)floor(factor * j))), std::min(largerWidth - 1, xs[0] + 1)},
						ys[2] = {std::max(0, std::min(largerHeight - 1, (int32_t)floor(factor * i))), std::min(largerHeight - 1, ys[0] + 1)};
					const float ax = clamp(factor * j - xs[0], 0.0f, 1.0f), ay = clamp(factor * i - ys[0], 0.0f, 1.0f);
					const double zs[2][2] =
					{
						{largerImg(ys[0], xs[0]), largerImg(ys[1], xs[0])},
						{largerImg(ys[0], xs[1]), largerImg(ys[1], xs[1])}
					};

#if 1
					/*
					get conn comps among the four pixels using four edges
					if x,y's conn comp has size >= 2, use it
					else if there's a conn comp w/ size >= 2, use it
					else use x,y's pixel
					*/
					const float maxDZ = .005; //at 1m; scaled by uncertainty
					const uint32_t xi = (x == xs[0]) ? 0 : 1, yi = (y == ys[0]) ? 0 : 1;
					std::vector<float> compDepths; //in x,y's comp
					compDepths.push_back(largerImg(y, x));
					if(fabs(largerImg(ys[yi], xs[!xi]) - largerImg(y, x)) < maxDZ * primesensor::stereoErrorRatio(largerImg(y, x))) compDepths.push_back(largerImg(ys[yi], xs[!xi]));
					if(fabs(largerImg(ys[!yi], xs[xi]) - largerImg(y, x)) < maxDZ * primesensor::stereoErrorRatio(largerImg(y, x))) compDepths.push_back(largerImg(ys[!yi], xs[xi]));
					if(fabs(largerImg(ys[!yi], xs[!xi]) - largerImg(y, x)) < maxDZ * primesensor::stereoErrorRatio(largerImg(y, x))) compDepths.push_back(largerImg(ys[!yi], xs[!xi]));
					if(compDepths.size() > 1)
					{
						float avg = 0;
						for(float z : compDepths) avg += z;
						foo(i, j) = avg / compDepths.size();
					}
					else
					{
						std::vector<float> otherCompDepths; //in the comp of the diagonally opposite pixel
						otherCompDepths.push_back(largerImg(ys[!yi], xs[!xi]));
						if(fabs(largerImg(ys[yi], xs[!xi]) - largerImg(ys[!yi], xs[!xi])) < maxDZ * primesensor::stereoErrorRatio(largerImg(ys[!yi], xs[!xi]))) otherCompDepths.push_back(largerImg(ys[yi], xs[!xi]));
						if(fabs(largerImg(ys[!yi], xs[xi]) - largerImg(ys[!yi], xs[!xi])) < maxDZ * primesensor::stereoErrorRatio(largerImg(ys[!yi], xs[!xi]))) otherCompDepths.push_back(largerImg(ys[!yi], xs[xi]));
						if(otherCompDepths.size() > 1)
						{
							float avg = 0;
							for(float z : otherCompDepths) avg += z;
							foo(i, j) = avg / otherCompDepths.size();
						}
						else
						{
							foo(i, j) = largerImg(y, x);
						}
					}
#else
					//bilerp ignoring invalid values: if there are any nearby valid values, use them

					float z_0, z_1, z;
					if(z00 > 0)
					{
						if(z10 > 0) z_0 = linterp(z00, z10, ax);
						else z_0 = z00;
					}
					else
					{
						if(z10 > 0) z_0 = z10;
						else z_0 = -1;
					}
					if(z01 > 0)
					{
						if(z11 > 0) z_1 = linterp(z01, z11, ax);
						else z_1 = z01;
					}
					else
					{
						if(z11 > 0) z_1 = z11;
						else z_1 = -1;
					}

					if(z_0 > 0)
					{
						if(z_1 > 0) z = linterp(z_0, z_1, ay);
						else z = z_0;
					}
					else
					{
						if(z_1 > 0) z = z_1;
						else z = -1;
					}

					foo(i, j) = z;
#endif
				}
				else foo(i, j) = -1;
			}

		ImPyramid[m] = foo;
	}
}
