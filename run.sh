#! /bin/bash

BASEDIR=/Users/gabrielvayaabad/Desktop/MDS/S2/MUD/D2

# convert datasets to feature vectors
echo "Extracting features..."
python3 extract-features.py $BASEDIR/MUD-D2/data/train/ > $BASEDIR/MUD-D2/train.feat $BASEDIR/MUD-D2/resources/DrugBank.txt $BASEDIR/MUD-D2/resources/HSDB.txt 
python3 extract-features.py $BASEDIR/MUD-D2/data/devel/ > $BASEDIR/MUD-D2/devel.feat $BASEDIR/MUD-D2/resources/DrugBank.txt $BASEDIR/MUD-D2/resources/HSDB.txt 

# train CRF model
echo "Training CRF model..."
python3 train-crf.py $BASEDIR/model.crf < $BASEDIR/MUD-D2/train.feat
# run CRF model
echo "Running CRF model..."
python3 predict.py $BASEDIR/model.crf < $BASEDIR/MUD-D2/devel.feat > $BASEDIR/MUD-D2/devel-CRF.out
# evaluate CRF results
echo "Evaluating CRF results..."
python3 evaluator.py NER $BASEDIR/MUD-D2/data/devel devel-CRF.out > $BASEDIR/MUD-D2/devel-CRF.stats


#Extract Classification Features
cat $BASEDIR/MUD-D2/train.feat | cut -f5- | grep -v ^$ > $BASEDIR/MUD-D2/train.clf.feat


# train Naive Bayes model
echo "Training Naive Bayes model..."
python3 train-sklearn.py $BASEDIR/model.joblib $BASEDIR/MUD-D2/vectorizer.joblib < $BASEDIR/MUD-D2/train.clf.feat
# run Naive Bayes model
echo "Running Naive Bayes model..."
python3 predict-sklearn.py $BASEDIR/model.joblib $BASEDIR/MUD-D2/vectorizer.joblib < $BASEDIR/MUD-D2/devel.feat > $BASEDIR/MUD-D2/devel-NB.out
# evaluate Naive Bayes results 
echo "Evaluating Naive Bayes results..."
python3 evaluator.py NER $BASEDIR/MUD-D2/data/devel devel-NB.out > devel-NB.stats

# remove auxiliary files.
rm train.clf.feat
