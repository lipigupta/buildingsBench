# Short-Term Load Forecasting Using Machine Learning

## Introduction

Short-Term Load Forecasting (STLF) is the process of predicting the electrical energy demand of residential and commercial buildings over short timeframes, ranging from the next hour to the next month [1][2][3]. STLF plays a vital role in efficient energy management [2][3], making it a key research area aligned with the United States government's goal of achieving energy dominance. We leverage the BuildingsBench platform, developed by the National Renewable Energy Laboratory, a U.S. Department of Energy lab, to explore the role of deep learning in STLF [1]. This project will provide hands-on experience in object-oriented programming, working with the PyTorch and Matplotlib libraries, tuning model parameters, and training models on graphics processing units (GPUs). It emphasizes practical implementation with minimal focus on theoretical foundations or extensive code development. Prior coding experience in Python is a plus, but no additional background is required.

## Pre-requisites

We assume you have experience using the command line, as well as some coding experience in at least one programming language. If you're not familiar with command line or Python, you should review introductory material on these topics before proceeding with the tutorials.

- Basic Unix: [https://github.com/olcf/hands-on-with-frontier/tree/master/challenges/Basic_Unix_Vim#basic-unix-and-vim-skills](https://github.com/olcf/hands-on-with-frontier/tree/master/challenges/Basic_Unix_Vim#basic-unix-and-vim-skills)  
- Intro to Python: [https://www.geeksforgeeks.org/python/python-programming-language-tutorial/](https://www.geeksforgeeks.org/python/python-programming-language-tutorial/)

## Repository Structure
<pre> 
BuildingsBenchTutorial/
│
├── Tutorials/
│   │
│   ├── Intro-Modules/
│   │   ├── Develop-Conda-Kernel.md
│   │   ├── Intro-Object-Oriented-Programming.ipynb
│   │   ├── Visualize-Time-Series-Data.ipynb
│   │   ├── Intro-Pytorch.ipynb    
│   │   ├── Intro-NN.ipynb
│   │   └── Readme.md
│   │
│   └── Final-Project-Modules/
│       ├── Explore-Dataset.ipynb
│       ├── Train-Model.ipynb
│       ├── Select-Model.ipynb
│       ├── Best-Overall-Model.md    
│       └── Readme.md    
│
├── Images/
│
├── BuildingsBench/
│    
└── README.md
</pre>

The `BuildingsBenchTutorial` repository contains three subdirectories: (1) `Images`, (2) `Tutorials`, and (3) `BuildingsBench`. You do not need to modify the `Images` or `BuildingsBench` directories. All development should be done within the `Tutorials` directory or directly from the terminal. However, to install the module, you will need to `cd` into the `BuildingsBench` directory and run the appropriate `pip` command. For more information, refer to Exercise 3 in the `Develop-Conda-Kernel.md` file when working on the Develop-Conda-Kernel tutorial.

The `Tutorials` directory contains two subdirectories: (1) `Intro-Modules` and (2) `Final-Project-Modules`. Please begin by reviewing the modules in the `Intro-Modules` directory, as they cover foundational concepts necessary for completing the final project notebooks in the `Final-Project-Modules` section.

## Expectations

The final project will be a collaborative effort. We expect each group member to contribute their best in completing their assigned section of the project. However, we understand that many participants may not have prior experience in programming or computer science. Therefore, please don’t feel stressed if you’re unable to complete all the tasks. Just do your best and support your team where you can. We encourage collaboration among team members, but we strongly discourage anyone from taking over or completing tasks assigned to other group members.

AI assistants like ChatGPT and Gemini are becoming more common in everyday work. Learning how to use AI effectively is becoming just as important as learning to code. Hence, using AI or online resources to aid your own creative and intellectual work is permitted. For example, using it to help you write a python function that you describe and design, as well as integrate yourself and test yourself. Do not run any AI-generated code without vetting it yourself. You are responsible for any code that you run on a system, regardless of if an AI model wrote it for you.

## Next Step: 
`BuildingsBenchTutorial/Tutorials/Intro-Modules/Readme.md`

## References

[1] Emami, Patrick, Abhijeet Sahu, and Peter Graf. *BuildingsBench: A Large-Scale Dataset of 900K Buildings and Benchmark for Short-Term Load Forecasting.* arXiv preprint arXiv:2307.00142 (2023).  
[2] Gross, George, and F. D. Galiana. *Short-Term Load Forecasting.* Proceedings of the IEEE, vol. 75, no. 12, 1987, pp. 1558–1573.  
[3] Akhtar, Saima, et al. *Short-Term Load Forecasting Models: A Review of Challenges, Progress, and the Road Ahead.* Energies, vol. 16, no. 10, 2023, p. 4060. [https://doi.org/10.3390/en16104060](https://doi.org/10.3390/en16104060)

## Authors

- Nrushad Joshi (ntj@ornl.gov)  
- Suzanne Parete-Koon, OLCF
- Lipi Gupta, NERSC
