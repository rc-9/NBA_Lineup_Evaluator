[![Issues][issues-shield]][issues-url]



<h1 align="center">NBA Lineup Evaluator</h1>
  <p align="center">
    University of Denver - Data Science Capstone
    <br />
    Romith Challa
    <br />
    <br />
    <a href="https://github.com/rc-9/NBA_Lineup_Evaluator">View Repo</a>
    ·
    <a href="https://github.com/rc-9/NBA_Lineup_Evaluator/issues">Report Bug</a>
    ·
    <a href="https://github.com/rc-9/NBA_Lineup_Evaluator/issues">Request Feature</a><br />
    <a href="https://ibb.co/kyXNGY0"><img src="https://i.ibb.co/kyXNGY0/z3.png" alt="z3" border="" /></a><br />
  </p>
</div>


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



## About The Project

<br />
Analytics has revolutionized the world of sports, with teams and their fans alike in constant pursuit of metrics to quantify player performances in new and insightful ways. This project aims to take advantage of NBA’s publicly available stats-API, in order to cluster players based on their specific contribution to a team’s offense and defense, in lieu of the outdated traditional position system. These new “sub-positions” are used in tandem with lineup data, in order to determine which combinations of players leads to “winning basketball”. This tool can be invaluable to any front-office or coaching staff as a data-driven guide for roster/lineup decisions.

<p align="right">(<a href="#top">back to top</a>)</p>



## Usage
<br/>

1. ```1_compiler.ipynb```: This notebook will employ webscraping techniques to retrieve player statistics and lineup data from NBA's stats-API. This information will then be filtered, compiled and outputted into a master CSV for the next stage of the project pipeline.
<br/>

2. ```2_processor.ipynb```: This notebook will process the retrieved NBA datasets through a series of cleaning and wrangling stages, based on initial explorations, to prepare for further stages down the project pipeline.
<br/>

3. ```3_explorer.ipynb```: This notebook will employ exploratory mining techniques to study the NBA players, discern statistical patterns and better inform the modeling stage of the project pipeline. **NOTE: Not all visual outputs are pre-loaded. For best visual output and to utilize the interactive toggle-menu for plots, execute this script in a Jupyter notebook (VS-Code does not fully support .ipynb interactive widgets).**
<br/>

4. ```4_modeler.py```: This notebook will conduct any remaining pre-processing steps and will execute unsupervised and supervised ML techniques on the players' statistical data.
<br/>

5. ```5_visualizer.py```: This script can be executed via command line to launch a preliminary dashboard application to assess lineups on their offensive and defensive synergy.

<p align="right">(<a href="#top">back to top</a>)</p>



## Acknowledgments

Data Science Capstone | Ritchie School of Engineering & Computer Science | University of Denver

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
[issues-shield]: https://img.shields.io/github/issues/rc-9/NBA_Lineup_Evaluator.svg?style=for-the-badge
[issues-url]: https://github.com/rc-9/NBA_Lineup_Evaluator/issues














