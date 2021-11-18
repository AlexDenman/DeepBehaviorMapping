# DeepBehaviorMapping
Code for Deep Behavior Mapping analysis, as reported in Zhang et al. 2021 (Neuron)
Requires Matlab R2021A

Provided here are the following:
* 4 MatLab functions that carry out the core steps of Deep Behavior Mapping
* A sample data file containing DeepLabCut tracking data (X and Y coordinates for 6 body parts) for a single mouse over 2 sessions
* A demo script which applies DBM to the sample data and visualizes the results

Body part X and Y coordinates are formatted as 1D cell arrays with one cell per training sequence (e.g. in the example data there are 57 trials, and therefore the input variables are 57x1 cell arrays). Each cell contains a single-precision floating point array with # of rows equal to # of tracked body parts, and # of columns equal to the sequence length (e.g. in the example data there are 6 tracked body parts and each sequence is 800 timestamps long, so each cell contains a 6x800 array).

The behavioral pseudolabels used as training targets (which are derived from subject position & event timing within the behavioral experiment, and therefore must be defined according to the experiment structure) are also formatted as 1D cell arrays with one cell per training sequence (N=57 in the example data). Each cell contains a 1x(number of timesteps) categorical array (1x800 in example data).

Core functions:
**dbmFormatInputs**
Accepts X and Y data for body parts, outputs transformed input sequences.

**dbmTrainModel**
Accepts formatted pose sequences from dbmFormatInputs, as well as training targets (behavior pseudolabels), and optional parameters. Returns trained network and training metadata.

**dbmGetModelOutputs**
Accepts trained network (from dbmTrainModel) and pose sequences (which may be the same sequences used to train the network, or any other sequences formatted with dbmFormatInputs with the same # of body parts). Returns pseudolabel predictions and LSTM activations, i.e. latent mapping of pose sequences as trajectories in behavior space.

**dbmExtractMicrostates**
Accepts latent trajectories (from dbmGetModelOutputs) and extracts a set of k microstates (k specified by experimenter), and provides microstate centroids. New data can be assigned to microstates by processing pose sequences with dbmFormatInputs / dbmGetModelOutputs, then finding the nearest microstate centroid for each timestep of the latent trajectories.
