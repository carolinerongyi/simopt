# Welcome to SimOpt!

SimOpt is a testbed of simulation-optimization problems and solvers. Its purpose is to encourage the development and constructive comparison of simulation-optimization (SO) solvers (algorithms). We are particularly interested in the finite-time performance of solvers, rather than the asymptotic results that one often finds in related literature.

The most-up-to-date publication about this library is [Eckman et al. (2021)](https://eckman.engr.tamu.edu/wp-content/uploads/sites/233/2022/01/SimOpt-software-paper.pdf).


For the purposes of this project, we define simulation as a very general technique for estimating statistical measures of complex systems. A system is modeled as if the probability distributions of the underlying random variables were known. Realizations of these random variables are then drawn randomly from these distributions. Each replication gives one observation of the system response, i.e., an evaluation of the objective function or stochastic constraints. By simulating a system in this fashion for multiple replications and aggregating the responses, one can compute statistics and use them for evaluation and design.

Several papers have discussed the development of SimOpt and experiments run on the testbed:
* [Pasupathy and Henderson (2006)](https://www.informs-sim.org/wsc06papers/028.pdf) explains the original motivation for the testbed.
* [Pasupathy and Henderson (2011)](https://www.informs-sim.org/wsc11papers/363.pdf) describes an earlier interface for MATLAB implementations of problems and solvers.
* [Dong et al. (2017)](https://www.informs-sim.org/wsc17papers/includes/files/179.pdf) conducts an experimental comparison of several solvers in SimOpt and analyzes their relative performance.
* [Eckman et al. (2019)](https://www.informs-sim.org/wsc19papers/374.pdf) describes in detail changes to the architecture of the MATLAB version of SimOpt and the control of random number streams.
* [Eckman et al. (2021)](https://eckman.engr.tamu.edu/wp-content/uploads/sites/233/2021/09/SimOpt-metrics-paper.pdf) introduces the design of experiments for comparing solvers; this design has been implemented in the latest Python version of SimOpt.




## Code
* The `master` branch contains the source code for the latest version of the testbed, which is written in Python.
* The `matlab` branch contains a previous stable version of the testbed written in MATLAB.

## Documentation
Full documentation for the source code can be found **[here](https://simopt.readthedocs.io/en/latest/index.html)**.

## Getting Started
The most straightforward way to interact with the library is to [clone](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) this repository. *(If you anticipate making improvements or contributions to SimOpt, you should first [fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo) the repository so that you can later request your changes be integrated via GitHub's pull request feature.)*

Download a copy of the cloned repository to your local machine and navigate to the `simopt` folder in your preferred integrated development environment (IDE). You will need to make sure that you have the following dependencies installed: Python 3, `numpy`, `scipy`, `matplotlib`, `pandas`, and `seaborn`. Run the command ``` pip install numpy scipy matplotlib pandas seaborn ``` to install them from the terminal.

The `demo` folder contains a handful of useful scripts that can be easily modified, as directed in the comments.

* `demo_model.py`: Run multiple replications of a simulation model and report its responses.

* `demo_problem.py`: Run multiple replications of a given solution for an SO problem and report its objective function values and left-hand sides of stochastic constraints.

* `demo_solver_problem.py`: Run multiple macroreplications of a solver on a problem, save the outputs to a .pickle file in the `experiments/outputs` folder, and save plots of the results to .png files in the `experiments/plots` folder.

* `demo_solvers_problems.py`: Run multiple macroreplications of groups of problem-solver pairs and save the outputs and plots.

* `demo_data_farming_model.py`: Create a design over model factors, run multiple replications at each design point, and save the results to a comma separated value (.csv) file in the `data_farming_experiments` folder.

* `demo_san-sscont-ironorecont_experiment`: Run multiple solvers on multiple versions of (s, S) inventory, iron ore, and stochastic activiy network problems and produce plots.

## Graphical User Interface (GUI) - User Guide

### Installation
To start up the GUI, navigate to the `simopt` directory and run the command ``` python3 GUI.py ``` from the terminal. The GUI depends on Python 3, `numpy`, `scipy`, `matplotlib`, `Pillow`, and `tkinter`. Run the command ``` pip install numpy scipy matplotlib Pillow tkinter ``` to install them from the terminal.

### Overview
From the GUI's main page, a user can create a specified **problem-solver pair** or a **problem-solver group**, run macroreplications, and generate plots.

The top of the main page provides three ways to create or continue working with an existing **problem-solver pair**:

1. Create an individual **problem-solver pair**.
2. Load a .pickle file of a previously created **problem-solver pair**.
3. Generate a cross-design **problem-solver group**.

At the bottom of the main page, there is a frame containing all **problem-solver pair**s and **problem-solver group**s. The first tab lists the **problem-solver pair**s ready to be run or post-replicated, the second tab lists the **problem-solver group**s made from the cross-design or by generating a **problem-solver group** from partial set of **problem-solver pair** in the first tab, and the third tab lists those **problem-solver pair**s that are ready to be post-normalized and prepared for plotting.

### Adding **problem-solver pair**s
This section explains how to add **problem-solver pair**s to the queue.

#### Loading a **problem-solver pair** from a file
1. In the top left corner, click "Load File". Your file system will pop up, and you can navigate to and select an appropriate \*.pickle file. (The GUI will throw an error if the selected file is not a \*.pickle file.)
2. Once a **problem-solver pair** object is loaded, it will be added to the "Queue of **problem-solver pair**s".
3. The Run and Post-Process buttons will be updated to accurately reflect whether the **problem-solver pair** has already been run and/or post-processed.

#### Creating a **problem-solver pair**
Instead of loading an existing **problem-solver pair**, you can create one from the main page of the GUI:
1. First, select a solver from the "Solver" dropdown list. Each of the solvers has an abbreviation for the type of problems the solver can handle. Once a solver is selected, the "Problem" list will be sorted and show only the problems that work with the selected solver.
2. Change factors associated with the solver as necessary.
3. All solvers with unique combinations of factors must have unique names, i.e., no two solvers can have the same name, but different factors. If you want to use the same solver twice for a problem but with different solver factors, make sure you change the name of the solver - the last solver factor - accordingly.
4. Select a problem from the "Problem" dropdown list.
Each problem has an abbreviation indicating which types of solver is compatible to solve it. The letters in the abbreviation stand for:
    <table>
        <tr>
          <th> Objective </th>
          <th> Constraint </th>
          <th> Variable </th>
          <th> Direct Gradient Observations </th>
        </tr>
        <tr>
          <td> Single (S) </td>
          <td> Unconstrained (U) </td>
          <td> Discrete (D) </td>
          <td> Available (G) </td>
        </tr>
      <tr>
          <td> Multiple (M) </td>
          <td> Box (B) </td>
          <td> Continuous (C) </td>
          <td> Not Available (N) </td>
        </tr>
      <tr>
          <td>  </td>
          <td> Deterministic (D) </td>
          <td> Mixed (M)  </td>
          <td>  </td>
        </tr>
      <tr>
          <td>  </td>
          <td> Stochastic (S) </td>
          <td> </td>
          <td>  </td>
        </tr>
    </table>

5. Change factors associated with the problem and model as necessary.
6. All problems with unique factors must have unique names, i.e., no two problems can have the same name, but different factors. If you want to use the same problem twice for a solver but with different problem or model factors, make sure you change the name of the problem - the last problem factor - accordingly.
7.  The number of macroreplications can be modified in the top-left corner. The default is 10.
8.  Select the "Add **problem-solver pair**" button, which only appears when a solver and problem is selected. The **problem-solver pair** will be added in the "Queue of **problem-solver pair**s."

#### Creating a **problem-solver group**
Currently, **problem-solver group**s can only be created within the GUI or command line; they cannot be loaded from a file. 

You can create a **problem-solver group** in two ways. The first is a "Cross-Design **problem-solver group**" that uses the default factors for a list of compatible problems and solvers. The second is creating a partial list of **problem-solver pair**s by selecting those from the "Queue of **problem-solver pair**s" and then clicking the "Make a **problem-solver group**" button. This will complete the cross-design for the partial list and create a new row in the "Queue of **problem-solver group**s".

By cross-designing a **problem-solver pair**, you can add a new item to the "Queue of **problem-solver group**s". 
1. Click the "Cross-Design **problem-solver group**" button.
2. Check the compatibility of the Problems and Solvers being selected. Note that solvers with deterministic constraint type can not handle problems with stochastic constraints (e.g., ASTRO-DF cannot be run on FACSIZE-2).
3. Specify the number of macroreplications - the default is 10.
4. Click "Confirm Cross-Design **problem-solver group**."
5. The pop-up window will disappear, and the **problem-solver pair**s frame will automatically switch to the "Queue of **problem-solver group**s".
6. To exit out of the **problem-solver group** pop-up without creating a **problem-solver group**, click the red "x" in the top-left corner of the window.


### Run a **problem-solver pair** or a **problem-solver group** 
To run a **problem-solver pair** or a **problem-solver group**, click the "Run" button in the "Queue of **problem-solver pair**s" or "Queue of **problem-solver group**s". Once the **problem-solver pair** or **problem-solver group** has been run, the "Run" button becomes disabled.
**Note:** Running a **problem-solver pair** can take anywhere from a couple of seconds to a couple of minutes depending on the **problem-solver pair** and the number of macroreplications.

### Post-Processing and Post-Normalization
Post-processing happens before post-normalizing and after the run is complete. You can specify the number of post-replications and the (proxy) optimal solution or function value.  After post-normalization is complete, the Plotting window appears.
To exit out of the Post-Process/ Normalize pop-up without post-processing or post-normalizing, click the red "x" in the top-left corner of the window.

#### - **problem-solver pair**
**problem-solver pair**s can be post-processed from the "Queue of **problem-solver pair**s" tab by clicking "Post-Process." Adjust Post-Processing factors as necessary. Only **problem-solver pair**s that have already been run and have not yet been post-processed can be post-processed. After post-processing, click the "Post-Normalize by Problem" tab to select which **problem-solver pair**s to post-normalize together.
* Only **problem-solver pair**s with the same problem can be post-normalized together.
* Once all **problem-solver pair**s of interest are selected, click the "Post-Normalize Selected" button at the bottom of the GUI (this button only appears when in the Post-Normalize tab).
* Update any values necessary and click "Post-Normalize" when the **problem-solver pair**s are ready to be post-normalized.

#### - **problem-solver group**
**problem-solver group**s are post-processed and post-normalized at the same time.
* Click the "Post-Process" button for the specific **problem-solver group**, then change any values necessary, then click "Post-Process".

### Plotting **problem-solver pair**s
The Plotting page is the same for both **problem-solver pair**s and **problem-solver group**s. Currently, multiple **problem-solver pair**s with the same problem can be plotted together, and any problem-solver pair from a single **problem-solver group** can be plotted together in Solvability Profiles, Difference Plots, and Area Scatter Plots. To return to the main page, click the red "x" in the top-left corner of the window.
1. On the left side, select one or more problems from the problem list.
2. Select solvers from the solver list.
3. On the right side, select a plot type and adjust plot parameters and settings.
There are 5 settings common to most plot types: Confidence Intervals, Number of Bootstrap Samples, Confidence Level, Plot Together, and Print Max HW.
The type of plots that are currently available in the GUI are: Mean Progress Curve, Quantile Progress Curve, Solve Time CDF, Scatter Plot, CDF Solvability, Quantile Solvability, CDF Difference Plot, Quantile Difference Plot, Terminal Box/Violin, and Terminal Scatter.
4. Click "Add."
5. All plots will show in the plotting queue, along with information about their parameters and where the file is saved.
6. To view one plot, click "View Plot." All plots can be viewed together by clicking "See All Plots" at the bottom of the page.

## Contributing
Users can contribute problems and solver to SimOpt by using [pull requests](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests) in GitHub or corresponding with the developers. The core development team currently consists of David Eckman (Texas A&M University), Shane Henderson (Cornell University), and Sara Shashaani (North Carolina State University).

## Citation
To cite this work, please use
```
@misc{simoptgithub,
  author = {D. J. Eckman and S. G. Henderson and S. Shashaani and R. Pasupathy},
  title = {{SimOpt}},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/simopt-admin/simopt}},
  commit = {eda24b9f6a5885a37321ad7f8534bf10dec22480}
}
```

## Acknowledgments
An earlier website for SimOpt, [http://www.simopt.org](http://www.simopt.org), was developed through work supported by the National Science Foundation under grant nos. DMI-0400287, CMMI-0800688 and CMMI-1200315.
Recent work on the development of SimOpt has been supported by the National Science Foundation under grant nos. DGE-1650441, CMMI-1537394, CMMI-1254298, CMMI-1536895, IIS-1247696, and TRIPODS+X DMS-1839346, by the Air Force Office of Scientific Research under grant nos. FA9550-12-1-0200, FA9550-15-1-0038, and FA9550-16-1-0046, and by the Army Research Office under grant no. W911NF-17-1-0094.
Any opinions, findings and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the National Science Foundation (NSF).
