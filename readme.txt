RGB-D Flow code
accompanying the ICRA 2013 paper "RGB-D Flow: Dense 3-D Motion Estimation Using Color and Depth"
-----
Evan Herbst
12 / 17 / 13
-----
based on public-domain code released in 2009 by Ce Liu

To build:

mkdir build
cd build
cmake ..
make
cd ..

To run the test driver, write depth maps to 16-bit png images (for example, create a cv::Mat_<uint16_t> and imwrite() it), then

bin/rgbdFlowDemo -i rgb0.png -d depth0.png -j rgb1.png -e depth1.png -t 6 -o outdir

This will write my .flo3 format. This distribution includes a function to read a .flo3 file into an opencv image; see optical_flow_utils/sceneFlowIO.h.

