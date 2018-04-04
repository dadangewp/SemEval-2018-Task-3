# SemEval-2018-Task-3---Irony-Detection

This system was built to deal with SemEval-2018 task 3. To be able to run this system, you need to configure several "stuffs".

Requirement
- numpy
- scipy
- scikit-learn
- nltk
- VADER
- Spyder IDE (optional)

Usage

1. Download the dataset on the Task official GitHub page (https://github.com/Cyvhee/SemEval2018-Task3).

2. Download several affective resources that used in this systems.
- Emolex (http://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm)
- EmoSenticNet (https://www.gelbukh.com/emosenticnet/)
- LIWC (http://www.liwc.net/download.php)
- DAL Normalized (Provided in this project)
- ANEW Normalized (Provided in this project)
- AFINN Normalized (Provided in this project)

3. Resolve several paths of file.
- Several paths in iodata folder.
- Paths of affective resources.

4. Run the program by executing main class.

To use this program, you can cite our work in :

@inproceedings{pamungkas2018nondicevosulserio,
  title={\#NonDicevoSulSerio: Irony Detection in English Tweets at Semeval 2018 Task 3},
  author={Pamungkas, Endang Wahyu and Patti, Viviana},
  booktitle={Proceedings of the 12nd International Workshop on Semantic Evaluation (SemEval 2018)},
  year={2018}
}