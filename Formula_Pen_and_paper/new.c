void CrossCheck(IntImage& seeds, FImage& seedsFlow, FImage& seedsFlow2, IntImage& kLabel2, int* valid, float th);
	float MatchCost(FImage& img1, FImage& img2, UCImage* im1f, UCImage* im2f, int x1, int y1, int x2, int y2);

	// a good initialization is already stored in bestU & bestV
	int Propogate(FImagePyramid& pyd1, FImagePyramid& pyd2, UCImage* pyd1f, UCImage* pyd2f, int level, float* radius, int iterCnt, IntImage* pydSeeds, IntImage& neighbors, FImage* pydSeedsFlow, float* bestCosts);
	void PyramidRandomSearch(FImagePyramid& pyd1, FImagePyramid& pyd2, UCImage* im1f, UCImage* im2f, IntImage* pydSeeds, IntImage& neighbors, FImage* pydSeedsFlow);
	void OnePass(FImagePyramid& pyd1, FImagePyramid& pyd2, UCImage* im1f, UCImage* im2f, IntImage& seeds, IntImage& neighbors, FImage* pydSeedsFlow);
	void UpdateSearchRadius(IntImage& neighbors, FImage* pydSeedsFlow, int level, float* outRadius);

	// minimum circle
	struct Point{
		double x, y;
	};
	double dist(Point a, Point b);
	Point intersection(Point u1, Point u2, Point v1, Point v2);
	Point circumcenter(Point a, Point b, Point c);
	// return the radius of the minimal circle
	float MinimalCircle(float* x, float*y, int n, float* centerX = NULL, float* centerY = NULL);

	//
	int _step;
	int _maxIters;
	float _stopIterRatio;
	float _pydRatio;

	int _isStereo;
	int _maxDisplacement;
	float _checkThreshold;
	int _borderWidth;