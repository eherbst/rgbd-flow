/*
 * recursiveMedianFiltering: of depth maps
 *
 * Evan Herbst
 * 8 / 27 / 12
 */

#include <cassert>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <boost/format.hpp>
#include <boost/pending/disjoint_sets.hpp>
#include <boost/unordered_map.hpp>
#include <boost/multi_array.hpp>
#include "rgbd_util/mathUtils.h"
#include "celiu_flow/recursiveMedianFiltering.h"
using std::vector;
using std::unordered_set;

namespace rgbd
{

/*
 * fill in invalid regions (values <= 0)
 *
 * windowHalfwidth: the nbrhood over which we take the median
 *
 * data: row-major and contiguous
 */
template <typename T>
void recursiveMedianFilterAux(const uint32_t width, const uint32_t height, T* const data, const int32_t windowHalfwidth)
{
	/*
	for pts by decreasing # known nbrs
		median filter w/ some window size
		update nbr counts of nbrs
	*/

	//list invalid pts by # of known-depth nbrs
	vector<unordered_set<uint32_t>> ptsByNumKnownNbrs(sqr(2 * windowHalfwidth + 1));
	vector<int32_t> numKnownNbrsByPt(height * width, -1); //-1 for pts with valid depth already
	uint32_t numPtsLeft = 0; //left to process
	for(int32_t i = 0, l = 0; i < height; i++)
		for(int32_t j = 0; j < width; j++, l++)
			if(data[l] <= 0)
			{
				uint32_t numKnownNbrs = 0;
				for(int32_t ii = std::max(0, i - windowHalfwidth); ii <= std::min((int32_t)height - 1, i + windowHalfwidth); ii++)
					for(int32_t jj = std::max(0, j - windowHalfwidth); jj <= std::min((int32_t)width - 1, j + windowHalfwidth); jj++)
					{
						const int ll = ii * width + jj;
						if(data[ll] > 0)
							numKnownNbrs++;
					}
				ptsByNumKnownNbrs[numKnownNbrs].insert(l);
				numKnownNbrsByPt[l] = numKnownNbrs;
				numPtsLeft++;
			}

	vector<float> vals(sqr(2 * windowHalfwidth + 1));
	while(numPtsLeft > 0)
	{
		int32_t maxNumNbrs = ptsByNumKnownNbrs.size() - 1;
		while(ptsByNumKnownNbrs[maxNumNbrs].empty() && maxNumNbrs >= 0) maxNumNbrs--;

		const uint32_t l = *ptsByNumKnownNbrs[maxNumNbrs].begin();
		const int32_t i = l / width, j = l % width;

		ptsByNumKnownNbrs[maxNumNbrs].erase(l);
		numKnownNbrsByPt[l] = -1;
		ASSERT_ALWAYS(maxNumNbrs > 0);
		uint32_t index = 0; //into vals
		for(int32_t ii = std::max(0, i - windowHalfwidth); ii <= std::min((int32_t)height - 1, i + windowHalfwidth); ii++)
			for(int32_t jj = std::max(0, j - windowHalfwidth); jj <= std::min((int32_t)width - 1, j + windowHalfwidth); jj++)
			{
				const int32_t ll = ii * width + jj;
				if(data[ll] > 0)
				{
					//do median filtering
					vals[index++] = data[ll];
				}
				else if(numKnownNbrsByPt[ll] >= 0)
				{
					//update list of unknown pts
					ptsByNumKnownNbrs[numKnownNbrsByPt[ll]].erase(ll);
					numKnownNbrsByPt[ll]++;
					ptsByNumKnownNbrs[numKnownNbrsByPt[ll]].insert(ll);
				}
			}
		std::sort(vals.begin(), vals.begin() + index);
		data[l] = vals[(int)floor(index / 2)];

		numPtsLeft--;
	}
}

void recursiveMedianFilter(const uint32_t width, const uint32_t height, float* const data, const int32_t windowHalfwidth)
{
	recursiveMedianFilterAux<float>(width, height, data, windowHalfwidth);
}
void recursiveMedianFilter(const uint32_t width, const uint32_t height, double* const data, const int32_t windowHalfwidth)
{
	recursiveMedianFilterAux<double>(width, height, data, windowHalfwidth);
}

} //namespace

/*
 * fill in invalid regions
 *
 * windowHalfwidth: the nbrhood over which we take the median
 */
void recursiveMedianFilter(DImage& depth, const int32_t windowHalfwidth)
{
	rgbd::recursiveMedianFilter(depth.width(), depth.height(), depth.data(), windowHalfwidth);
}

/*
 * fill in only small invalid regions
 *
 * we'll use the first three channels of col to get a color similarity measure; each channel should be in [0, 1)
 */
void recursiveMedianFilterSmallInvalidRegionsUsingColor(DImage& depth, const DImage& col, const uint32_t maxRegionSizeToFill)
{
//	static int calls = 0;
//	ImageIO::saveImage((boost::format("pre%1%.png") % calls).str().c_str(), depth.data(), depth.width(), depth.height(), depth.nchannels(), ImageIO::normalized);

	const int32_t windowHalfwidth = 1; //TODO ?

	/*
	 * do conn comps on invalid pixels
	 */
	typedef std::pair<int32_t, int32_t> ptT;
	typedef boost::associative_property_map<boost::unordered_map<ptT, uint32_t> > rankMapT;
	typedef boost::associative_property_map<boost::unordered_map<ptT, ptT> > parentMapT;
	boost::unordered_map<ptT, uint32_t> rankMap;
	boost::unordered_map<ptT, ptT> parentMap;
	boost::disjoint_sets<rankMapT, parentMapT> sets(boost::make_assoc_property_map(rankMap), boost::make_assoc_property_map(parentMap));
	for(int32_t i = 0; i < depth.height(); i++)
		for(int32_t j = 0; j < depth.width(); j++)
			if(depth(i, j) <= 0)
				sets.make_set(std::make_pair(j, i));
	for(int32_t i = 0; i < depth.height(); i++)
		for(int32_t j = 0; j < depth.width(); j++)
			if(depth(i, j) <= 0)
			{
				if(i > 0 && depth(i - 1, j) <= 0) sets.union_set(std::make_pair(j, i), std::make_pair(j, i - 1));
				if(j > 0 && depth(i, j - 1) <= 0) sets.union_set(std::make_pair(j, i), std::make_pair(j - 1, i));
			}
	std::unordered_map<ptT, std::vector<ptT>> ptsByComp;
	for(auto p : rankMap)
		ptsByComp[sets.find_set(p.first)].push_back(p.first);

	/*
	 * filter each small enough region
	 */
	for(const auto& c : ptsByComp)
		if(c.second.size() <= maxRegionSizeToFill)
		{
			/*
			 * list invalid pts by # of known-depth nbrs
			 */
			vector<unordered_set<ptT>> ptsByNumKnownNbrs(sqr(2 * windowHalfwidth + 1));
			boost::multi_array<int32_t, 2> numKnownNbrsByPt(boost::extents[depth.height()][depth.width()]); //-1 for pts with valid depth already
			std::fill(numKnownNbrsByPt.data(), numKnownNbrsByPt.data() + numKnownNbrsByPt.num_elements(), -1);
			uint32_t numPtsLeft = 0; //left to process
			for(ptT p : c.second)
			{
				const int32_t i = p.second, j = p.first;
				uint32_t numKnownNbrs = 0;
				for(int32_t ii = std::max(0, i - windowHalfwidth); ii <= std::min((int32_t)depth.height() - 1, i + windowHalfwidth); ii++)
					for(int32_t jj = std::max(0, j - windowHalfwidth); jj <= std::min((int32_t)depth.width() - 1, j + windowHalfwidth); jj++)
						if(depth(ii, jj) > 0)
							numKnownNbrs++;
				ptsByNumKnownNbrs[numKnownNbrs].insert(p);
				numKnownNbrsByPt[p.second][p.first] = numKnownNbrs;
				numPtsLeft++;
			}

			while(numPtsLeft > 0)
			{
				int32_t maxNumNbrs = ptsByNumKnownNbrs.size() - 1;
				while(ptsByNumKnownNbrs[maxNumNbrs].empty() && maxNumNbrs >= 0) maxNumNbrs--;

				const ptT p = *ptsByNumKnownNbrs[maxNumNbrs].begin();
				const int32_t i = p.second, j = p.first;

				ptsByNumKnownNbrs[maxNumNbrs].erase(p);
				numKnownNbrsByPt[i][j] = -1;
				ASSERT_ALWAYS(maxNumNbrs > 0);
				float zSum = 0, weightSum = 0;
				for(int32_t ii = std::max(0, i - windowHalfwidth); ii <= std::min((int32_t)depth.height() - 1, i + windowHalfwidth); ii++)
					for(int32_t jj = std::max(0, j - windowHalfwidth); jj <= std::min((int32_t)depth.width() - 1, j + windowHalfwidth); jj++)
					{
						const ptT p2 = std::make_pair(jj, ii);
						if(depth(ii, jj) > 0)
						{
							//do filtering
							const float w = exp(-(sqr(col(i, j, 0) - col(ii, jj, 0)) + sqr(col(i, j, 1) - col(ii, jj, 1)) + sqr(col(i, j, 2) - col(ii, jj, 2))) / .6/* TODO ? */);
							zSum += depth(ii, jj) * w;
							weightSum += w;
						}
						else if(numKnownNbrsByPt[ii][jj] >= 0)
						{
							//update list of unknown pts
							ptsByNumKnownNbrs[numKnownNbrsByPt[ii][jj]].erase(p2);
							numKnownNbrsByPt[ii][jj]++;
							ptsByNumKnownNbrs[numKnownNbrsByPt[ii][jj]].insert(p2);
						}
					}
				depth(i, j) = zSum / weightSum;
				numPtsLeft--;
			}
		}

//	ImageIO::saveImage((boost::format("post%1%.png") % calls).str().c_str(), depth.data(), depth.width(), depth.height(), depth.nchannels(), ImageIO::normalized);
//	calls++;
}
