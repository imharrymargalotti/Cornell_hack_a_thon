# Cornell_hack_a_thon
#Colonosco-PY
## Contributing:
* Harrison Margalotti
* Tim Clerico
* Connor Robinson
* Jeff Page
* Erika Ganda
## Description:
Project built in about 36 hours for the Cornell 2019 Micobiome Hackathong where we implemented a machine learning algorithm to classify someone as having colorectal cancer or not based on the bacterial microbiome from their poop. We did this through anomaly detection (isolation forest) across 3 studies; one from the US, one from China and one from Germany and found out that if by training  the anomaly detector on just the healthy people from China that we can still distinguish healthy and diseased people from Germany and the US meaning that they have similar healthy microbiomes. 

Then we used a classification algorithm called random forest where we trained it on just China and scored our predictions on US and Germany with varying levels of accuracy. When then trained the classifier on data from China and US pooled together, and ranked Germany, it received the exact same score concluding that there are underlying bacteria present in all 3 places that are relevant to the cancer. 

We concluded by finding out which bacteria had the greatest impact on score (by shuffling one feature (bacteria) at a time and measureing how it impacted the score) and we able to identify the top 5 bacteria’s that exist in those microbiomes that have some impact on the absence or presence of colorectal cancer and we backed it up with a study that matched our conclusion. 

Our algorithm was able to lead us to these findings without including any risk factors: age, sex, obesity, other diseases...we worked exclusively with bacteria samples

### Prerequisites

requirements.txt (included)
data in .npz files (included)

## Built With
* Python 3.7
* SciKit-Lean
* NumPy
## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

### Acknowledgements
* Andrew 'the machine man' Ng
* Juan Felipe Beltran
