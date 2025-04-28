# Inflection Point-Based Trajectory Fingerprinting for Clustering and Anomaly Detection
> <div align="justify">Identifying unsafe personal watercraft behavior is critical for maritime safety, especially in crowded recreational areas. This paper presents a novel, lightweight, scalable method for trajectory clustering and anomaly detection based on inflection point sequences. Identifying patterns is challenging due to varying trajectory lengths and environments. By encoding trajectory shape changes into compact fingerprints, the method avoids the computational burdens of traditional techniques. Results indicate that the proposed fingerprinting method can effectively differentiate normal from anomalous driving patterns, offering a viable solution for real-time anomaly detection. This study contributes a simple yet robust trajectory analysis framework applicable to maritime monitoring and broader trajectory mining applications, paving the way for improved safety and real-time decision support systems. Trajectories are segmented, inflection points are extracted, and clustering is performed using trajectory fingerprints. Human evaluation experiments validate the approach, demonstrating that a larger segment window size enhances clustering accuracy and alignment with expert judgment.</div>

## Environment setup &amp; activation
Use `conda` and `inflection_points.yml` to create an environment.

    cd <cloning directory>
    conda env create -f inflection_points.yml
    conda activate inflection_points

## Important notes
- The code should be run from the root cloning directory.
- If you want to generate fingerprints and cluster randomly created trajectories, modify the parameters when running the script ```trajectories_clustered.py```.
- Trajectory generation time varies with the model used, but usually takes about 0.16 seconds for a single trajectory with 40 points (the default settings).
- The fingerprinting process identifying inflection points in trajectory segments takes about 3 miliseconds for a trajectory with 40 points and a window size of 20 (the default settings).
- The DBSCAN and K-means clustering take about 0.3 seconds for 1167 trajectories with 40 points and a window size of 20 (the default settings).
- Saving plots of inflection points for each trajectory significantly increases execution time and size on disk, so this is disabled by default, but can be included if desired.
- The results of an experiment run with the same settings will overwrite the existing data.
- Probability density distribution for random trajectory generation are the result of previous research by the authors on Bayesian and Markov chain models for trajectory forecasting.

## Script parameters

### Running the script:

    python trajectories_clustered.py --trajectories <trajectories to generate> --points <trajectory length> --window <window size> --features <number of clustering features> --neighbors <minimum DBSCAN cluster size> --clusters <number of K-means clusters> --plots <generate inflection point plots> --verbose <fingerprint command line output>
    
                   ┌────────┐
    points ───────►│features├─────► clusters
                   └────────┘

#### trajectories

Defining the number of trajectories, the default value is 1167. If the number of trajectories is not an integer greater than 0, trajectories are not generated. Trajectories in .csv files are stored in a .csv file in the "trajectories" subdirectory.

#### points

Defining the number of longitude and latitude points, the default value is 40. If the number of longitude and latitude points is not an integer greater than 0, the default value is applied. Trajectory generation execution times are stored in a .csv file in the "results" subdirectory.

#### window

Defining the window size, the default value is 20. If the window size is not an integer greater than 0, the default value is applied. If the window size is larger than the number of longitude and latitude points, the window size is decreased to match the mumber of longitude and latitude points. Trajectory features with the specified window size are stored in a .csv file in the "results" subdirectory. Trajectory fingerprinting execution times with the specified window size are stored in a .csv file in the "results" subdirectory.

#### features

Defining the maximum number of features for clustering, the default value is 200. Clustering outcomes for the specified window size and algorithm are stored in a .csv file in the "results" subdirectory. Clustering outcomes for the specified window size and algorithm are visualised using PCA and TSNE, with .pdf, .png and .svg plots stored in the "plots" subdirectory. Trajectory features with PCA and TSNE for the specified window size and algorithm are stored in a .csv file in the "results" subdirectory.

#### neighbors

Defining the minimum cluster size for DBSCAN clusters, the default value is 346. If the value is not an integer greater than 0 and less than or equal to the number of trajectories, DBSCAN clustering is not applied.

#### clusters

Defining the number of clusters for KMeans clustering, the default value is 2. If the value is not an integer greater than 1, KMeans clustering is not applied.

#### plots

The user can decide if they want to generate a plot showing the inflection points for each trajectory. The default value is false, so the plots will not be generated unless the parameter is specified. Plots showing the inflection points in .pdf, .png and .svg formats are stored in the "trajecotry_plots" subdirectory.

#### verbose

The user can decide if they want to read the output of trajectory fingerprinting in the command line. The default value is false, so only the clustering results will be printed in the command line.