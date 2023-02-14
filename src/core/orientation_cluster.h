class OrientationCluster {
    EulerSearch                                               search;
    float                                                     PsiStep;
    int                                                       number_of_clusters;
    std::vector<std::vector<std::tuple<float, float, float>>> clusters;

    OrientationCluster(EulerSearch& search, float PsiStep);
};