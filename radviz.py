# pylint: disable=relative-beyond-top-level
import numpy as np
import itertools
import math
from .kendall_tau import KendalTauDistance
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import davies_bouldin_score as DBIndex
from scipy.stats import rankdata
dtype = np.float64
#
#
#
#
#
class RadVizPermutations:
    """
    (n-1)! / 2 permutazioni
    """
    def __init__(self, nDim, dtype=np.uint8):
        self.dtype = dtype
        self.nDim = nDim
        self.n = int(math.factorial(nDim-1)/2)

        p = np.array(list(itertools.permutations(np.arange(1, nDim, dtype=np.uint8))))
        def _fn(x):
            if x[-1] < x[0]: return x[::-1]
            else: return x
        p = np.apply_along_axis(_fn, 1, p)
        p = np.lib.arraysetops.unique(p, axis=0)
        self.generator = p
        
    def get(self, start=0, stop=None):
        if stop is None:
            stop = self.n-1
        permutations = np.zeros((stop-start+1, self.nDim), dtype=np.uint8)
        permutations[:,range(1,self.nDim)] = self.generator[start:stop+1,]
        return permutations

    @staticmethod
    def count(nDim):
        return int(math.factorial(nDim-1)/2)
#
#
#
#
class RadViz:
    """
    A class representing a RadViz
    """
    def __init__(self, dataset, permutation):
        self.nDim = dataset.nDim
        self.dataset = dataset
        self.permutation = permutation
        
        self.points = self._compute_points(dataset.rowNormalizedData[:,self.permutation], True)
        self.anchors = self._compute_anchors()
        self.anchorDistances = self._compute_anchorDistances()
        self.points_distanceMatrix = euclidean_distances(self.points)
        
        self.distances = np.sqrt(np.square(self.points[:,0]) + np.square(self.points[:,1]), dtype=dtype)
        self.dbindex = self._compute_dbindex()
        
        self.error_eff = self._compute_errorEffectiveness()
        self.error_dd, self.error_ddn, self.error_pr, self.error_prn = self._compute_error_data()

        self.dependent_da = self._compute_dependent_da()
        self.independent_da = self.compute_independent_da()
        self.radviz_pp = self._compute_radviz_pp()


        self.clumping_50 = 1 - np.quantile(self.distances, 0.50, interpolation="linear")
        self.clumping_75 = 1 - np.quantile(self.distances, 0.75, interpolation="linear")
        self.density = self._compute_density()

        self.error_np, self.error_npn, self.error_ns, self.error_nsn, self.error_pps, self.error_ppsn = self._compute_error_neighbors()
        

    def getMetrics(self):
        global_metrics = {
            "avg_distances": np.mean(self.distances),
            "clumping_50": self.clumping_50,
            "clumping_75": self.clumping_75,
            "density": self.density,
            "avg_error_eff": np.mean(self.error_eff),
            "dbindex": self.dbindex,
            "error_dd": self.error_dd,
            "error_ddn": self.error_ddn,
            "error_pr": self.error_pr,
            "error_prn": self.error_prn,
            "error_np": self.error_np, 
            "error_npn": self.error_npn, 
            "error_ns": self.error_ns, 
            "error_nsn": self.error_nsn,
            "error_pps": self.error_pps, 
            "error_ppsn": self.error_ppsn,
            "dependent_da": self.dependent_da,
            "independent_da": self.independent_da,
            "radviz_pp": self.radviz_pp
        }
        points_metrics = {
            "distances": self.distances,
            "error_eff": self.error_eff,
        }
        return points_metrics, global_metrics
    

    def getMetricsNames(self):
        pm, gm = self.getMetrics()
        return sorted(list(pm.keys())), sorted(list(gm.keys()))

    def getMetricsArray(self):
        gm, sm = self.getMetrics()

        global_keys = self.getMetricsNames()
        global_array = np.full(len(global_keys), np.nan)
        for i in enumerate(gm):
            global_array[i] = gm[global_keys[i]]

        points_keys = self.getMetricsNames()
        points_array = np.full(len(points_keys, len(self.points)), np.nan)
        for i in enumerate(sm):
            points_array[i] = sm[points_keys[i]]
        
        return points_array, global_array
    #
    #
    #
    #
    @staticmethod
    def normalizedRows(X):
        """
        normalizes each row by its sum
        """
        s = np.sum(X, axis=1)
        s = np.where(s==0, 1, s)
        nData = np.array(X / s[:, np.newaxis])
        return nData
    
    @staticmethod
    def stress(L,C):
        """
        stress function
        """
        N = np.sum(np.square(L - C))
        D = np.sum(np.square(C))
        return math.sqrt(N/D)

    @staticmethod
    def cdist(i,j, nDim):
        """
        circular distance between dimension anchors
        """
        d1 = 0
        d2 = 0
        k = i
        while k != j:
            d1 +=1
            k = (k+1)%nDim
        k = i
        while k != j:
            d2 +=1
            k = (k-1)%nDim
        
        return min(d1, d2)
    
    
    def _compute_points(self, normalizedData, rowNormalized=False):
        rowNormalizedData = normalizedData
        if not rowNormalized:
            rowNormalizedData = RadViz.normalizedRows(normalizedData)
        nDim = rowNormalizedData.shape[1]
        radvizAngle = 2 * np.pi / nDim
        xCoeff = np.array([ np.cos(radvizAngle*i) for i in range(nDim)])
        yCoeff = np.array([ np.sin(radvizAngle*i) for i in range(nDim)])
        X = np.sum(rowNormalizedData*xCoeff, axis=1, dtype=dtype)[:, np.newaxis]
        Y = np.sum(rowNormalizedData*yCoeff, axis=1, dtype=dtype)[:, np.newaxis]
        points = np.hstack([X, Y])
        return points
    
    def _compute_dbindex(self):
        res = np.nan
        if self.dataset.classes is not None: 
            res = DBIndex(self.points, self.dataset.classes)
        return res

    def _compute_anchors(self):
        """
        anchors coordinates
        """
        angle = 2*np.pi/self.nDim
        anchors = np.array([ [np.cos(j*angle), np.sin(j*angle)] for j in range(self.nDim) ], dtype=dtype)
        return anchors

    def _compute_anchorDistances(self):
        """
        computes distance from anchors for each point
        """
        dist = euclidean_distances(self.points, self.anchors)
        return dist
    
    def _compute_errorEffectiveness(self):
        Xn = self.dataset.normalizedData[:,self.permutation]
        A = self.anchorDistances
        ind = np.fliplr( np.argsort(A, axis=1) ) #indici che ordinano le distanze ordine decrescente -> a distanza minima voglio valore massimo
        def fn(e): ##riordino x[i,] secondo gli indici in ind[i,]
            x, i = np.array_split(e,2)
            return x[ np.array(i, dtype=int)]
        Xn = np.apply_along_axis(fn, 1, np.c_[Xn, ind])
        
        e = np.apply_along_axis(KendalTauDistance.diff_distance_norm, 1, Xn)
        return e


    def _compute_error_data(self):
        L = RadViz.normalizedRows(self.points_distanceMatrix)
        C = RadViz.normalizedRows(self.dataset.data_distanceMatrix)
        Cn = RadViz.normalizedRows(self.dataset.normalizedData_distanceMatrix)
        
        error_dd = RadViz.stress(L, C)
        error_ddn = RadViz.stress(L, Cn)
        error_pr = np.square(error_dd)
        error_prn = np.square(error_ddn)
        return error_dd, error_ddn, error_pr, error_prn

   
    def compute_independent_da(self):
        N = np.full((self.nDim, self.nDim), 0, dtype=float)
        for i in range(self.nDim):
            for j in range(self.nDim):
                N[i,j] = 1 - ( RadViz.cdist(i,j,self.nDim) / (self.nDim/2) )
        S = self.dataset.getFeaturesSimilarity(self.permutation)
        return np.sum(N*S)
    
    def _compute_dependent_da(self):
        featuresPoints = self._compute_points(self.dataset.getFeaturesSimilarity(self.permutation)) 
        distances = euclidean_distances(featuresPoints, self.anchors)
        return np.sum(distances)
   

    def _compute_radviz_pp(self):
        N = np.full((self.nDim, self.nDim), 0, dtype=float)
        for i in range(self.nDim):
            for j in range(self.nDim):
                N[i,j] = 1 - (RadViz.cdist(i,j,self.nDim) / (self.nDim/2) )
        C = self.dataset.getFeaturesCorrelation(self.permutation)
        return np.sum(N*C)
    
    def _compute_density(self):
        res = 0
        distances = self.distances
        n = 100
        r = np.linspace(0, 1, n+1)
        for i in range(n):
            area = (np.pi * np.square(r[i+1])) - (np.pi * np.square(r[i]))
            pointsNumber = len( (np.where(distances >= r[i]) and np.where(distances < r[i+1]))[0] )
            res += (pointsNumber / area)
        res = (res/n)
        return res
    
    def _compute_error_neighbors(self):
        k = int(np.sqrt(self.points.shape[0]))
        radvizRank = np.apply_along_axis(rankdata, 1, self.points_distanceMatrix)
        radvizNeighbors = np.argsort(radvizRank, 1)
        #
        #
        #
        originalNeighbors_set = np.apply_along_axis(set, 1, self.dataset.neighbors[:,0:k])
        originalNormalizedNeighbors_set = np.apply_along_axis(set, 1, self.dataset.normalizedNeighbors[:,0:k])
        radvizNeighbors_set = np.apply_along_axis(set, 1, radvizNeighbors[:,0:k])

        def jaccard_diss(x):
            union = len(x[0].union(x[1]))
            if union == 0: return 1
            intersection = len(x[0].intersection(x[1]))
            return 1 - (intersection/union)
        error_np = np.mean( np.apply_along_axis(jaccard_diss, 1, np.c_[originalNeighbors_set, radvizNeighbors_set]) )
        error_npn = np.mean( np.apply_along_axis(jaccard_diss, 1, np.c_[originalNormalizedNeighbors_set, radvizNeighbors_set]) )
        #
        #
        #
        ns_array = np.full(self.points.shape[0], 0, dtype=dtype)
        nsn_array = np.full(self.points.shape[0], 0, dtype=dtype)
        for i in range(self.points.shape[0]):
            for j in range(k):
                j_original = self.dataset.neighbors[i,j]
                j_radviz = radvizNeighbors[i,j]
                ns_array[i] += 0.5*(k-j)*np.abs(radvizRank[i,j_radviz] - self.dataset.neighborsRank[i,j_radviz])
                ns_array[i] += 0.5*(k-j)*np.abs(radvizRank[i,j_original] - self.dataset.neighborsRank[i,j_original])
        for i in range(self.points.shape[0]):
            for j in range(k):
                j_original = self.dataset.normalizedNeighbors[i,j]
                j_radviz = radvizNeighbors[i,j]
                nsn_array[i] += 0.5*(k-j)*np.abs(radvizRank[i,j_radviz] - self.dataset.normalizedNeighborsRank[i,j_radviz])
                nsn_array[i] += 0.5*(k-j)*np.abs(radvizRank[i,j_original] - self.dataset.normalizedNeighborsRank[i,j_original])
        
        error_ns = np.mean(ns_array)
        error_nsn = np.mean(nsn_array)
        #
        #
        #
        pps_array = np.full(self.points.shape[0], 0, dtype=dtype)
        ppsn_array = np.full(self.points.shape[0], 0, dtype=dtype)
        for i in range(self.dataset.neighbors.shape[0]):
            d = self.dataset.data_distanceMatrix[i, self.dataset.neighbors[i, 0:k]]
            dn = self.dataset.normalizedData_distanceMatrix[i, self.dataset.normalizedNeighbors[i, 0:k]]
            r = self.points_distanceMatrix[i, self.dataset.neighbors[i, 0:k]]
            norm_d = np.linalg.norm(d)
            norm_dn = np.linalg.norm(dn)
            norm_r = np.linalg.norm(r)
            pps_array[i] = np.linalg.norm( (d/norm_d) - (r/norm_r) )
            ppsn_array[i] = np.linalg.norm( (dn/norm_dn) - (r/norm_r) )
        
        error_pps = np.mean(pps_array)
        error_ppsn = np.mean(ppsn_array)
        #
        #
        #
        return error_np, error_npn, error_ns, error_nsn, error_pps, error_ppsn




