#! /usr/bin/python3

import sys
import re
from os import listdir
import string

from xml.dom.minidom import parse
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet


   
## --------- tokenize sentence ----------- 
## -- Tokenize sentence, returning tokens and span offsets

def tokenize(txt):
    offset = 0
    tks = []
    ## word_tokenize splits words, taking into account punctuations, numbers, etc.
    for t in word_tokenize(txt):
        ## keep track of the position where each token should appear, and
        ## store that information with the token
        offset = txt.find(t, offset)
        tks.append((t, offset, offset+len(t)-1))
        offset += len(t)

    ## tks is a list of triples (word,start,end)
    return tks


## --------- get tag ----------- 
##  Find out whether given token is marked as part of an entity in the XML

def get_tag(token, spans) :
   (form,start,end) = token
   for (spanS,spanE,spanT) in spans :
      if start==spanS and end<=spanE : return "B-"+spanT
      elif start>=spanS and end<=spanE : return "I-"+spanT

   return "O"

# Function that gets de POS tag of a token, for lemmatization. 
def get_wordnet_pos(treebank_tag):
   if treebank_tag.startswith('J'):
      return wordnet.ADJ
   elif treebank_tag.startswith('V'):
      return wordnet.VERB
   elif treebank_tag.startswith('N'):
      return wordnet.NOUN
   elif treebank_tag.startswith('R'):
      return wordnet.ADV
   else:
      return wordnet.NOUN  # Default to noun if POS tag is not recognized
 
## --------- Feature extractor ----------- 
## -- Extract features for each token in given sentence

def extract_features(tokens, drugs, brands, groups, drug_names) :

   # for each token, generate list of features and add it to the result
   result = []
   pos_tags = pos_tag([t[0] for t in tokens]) # get POS tags of the whole sentence
   
   for k in range(0,len(tokens)):
      tokenFeatures = []
      t = tokens[k][0]

      ##################################Form##################################### YES
      tokenFeatures.append("form="+t)
      tokenFeatures.append("lowerform="+t.lower())
      tokenFeatures.append("length="+str(len(str(t))))
      tokenFeatures.append("suf3="+t[-3:])
      tokenFeatures.append("suf5="+t[-5:])
      tokenFeatures.append("pre3="+t[:3])
      tokenFeatures.append("pre5="+t[:5])

      ##############################POSTagging#################################### NO
      '''pos = pos_tags[k][1]
      tokenFeatures.append("postag="+pos_tags[k][1])'''

      ##############################Lemmatizer#################################### NO
      '''lemmatizer = WordNetLemmatizer()
      pos = pos_tags[k][1]
      lemma = lemmatizer.lemmatize(t, pos=get_wordnet_pos(pos))
      tokenFeatures.append("lemma="+lemma)'''

      #############################Dashes and numbers############################# YES
      if re.search(r'[0-9-]', t):
         if re.search(r'[a-zA-Z]', t):
            tokenFeatures.append("has_spcar=Some")
         else:
            tokenFeatures.append("has_spcar=All")
      else:
         tokenFeatures.append("has_spcar=None")
      
      ###############################Numbers###################################### NO
      '''if re.search(r'\d', t):
         if re.search(r'[a-zA-Z]', t):
            tokenFeatures.append("has_num=Some")
         else:
            tokenFeatures.append("has_num=All")
      else: 
         tokenFeatures.append("has_num=None")'''

      ###############################Dashes######################################## NO
      '''if '-' in t:
         tokenFeatures.append("dashes=Yes")
      else:
         tokenFeatures.append("dashes=No")'''

      ##############################Capital patterns############################### YES
      if t.isupper():
         tokenFeatures.append("cap=All")
      elif t[0].isupper():
         tokenFeatures.append("cap=Ini")
      else:  
         tokenFeatures.append("cap=None")

      ###############################Ends in s##################################### YES
      if t[-1] == 's':
         tokenFeatures.append("is_plural=Yes")
      else:
         tokenFeatures.append("is_plural=No")
      
      ################################Resources#################################### YES
      if t.lower() in drugs:
         tokenFeatures.append("resource=Drug")
      elif t.lower() in drug_names:
         tokenFeatures.append("resource=Drug_n")
      elif t.lower() in brands:
         tokenFeatures.append("resource=Brand")
      elif t.lower() in groups:
         tokenFeatures.append("resource=Group")
      else:
         tokenFeatures.append("resource=None")
      
      ###############################Punctuation################################### YES
      punct = False
      for char in t:
         if char in string.punctuation:
            tokenFeatures.append("punct=Yes")
            punct = True
      if not punct:
         tokenFeatures.append("punct=No")

      ##############################Previous token##################################
      if k>0:
         tPrev = tokens[k-1][0]
         tokenFeatures.append("formPrev="+tPrev)
         #tokenFeatures.append("lowerformPrev="+tPrev.lower()) NO
         #tokenFeatures.append("lengthPrev="+str(len(str(tPrev)))) NO
         tokenFeatures.append("suf3Prev="+tPrev[-3:])
         tokenFeatures.append("suf5Prev="+tPrev[-5:])
         tokenFeatures.append("pre3Prev="+tPrev[:3])
         tokenFeatures.append("pre5Prev="+tPrev[:5])
         #tokenFeatures.append("postagPrev="+pos_tags[k-1][1]) NO

         ###########Features not included in previous token##############
         '''if tPrev.isupper():
            tokenFeatures.append("capPrev=All")
         elif tPrev[0].isupper():
            tokenFeatures.append("capPrev=Ini")
         else:  
            tokenFeatures.append("capPrev=None")'''
         '''if re.search(r'[0-9-]', tPrev):
            if re.search(r'[a-zA-Z]', tPrev):
               tokenFeatures.append("has_spcarP=Some")
            else:
               tokenFeatures.append("has_spcarP=All")
         else:
            tokenFeatures.append("has_spcarP=None")'''
         '''punct = False
         for char in tPrev:
            if char in string.punctuation:
               tokenFeatures.append("punctP=Yes")
               punct = True
         if not punct:
            tokenFeatures.append("punct=No")'''
         
      else: 
         tokenFeatures.append("BoS")

      #############################2-previous token################################
      '''if k>1 and k!= 0:
         tPrev = tokens[k-2][0]
         tokenFeatures.append("formPrev2="+tPrev)
         tokenFeatures.append("suf3Prev2="+tPrev[-3:])
         tokenFeatures.append("pre3Prev2="+tPrev[:3])
      else:
         tokenFeatures.append("formPrev2=None")
         #tokenFeatures.append("cap=False")'''
      

      #################################Next token###################################
      if k<len(tokens)-1 :
         tNext = tokens[k+1][0]
         tokenFeatures.append("formNext="+tNext)
         #tokenFeatures.append("lowerformNext="+tNext.lower()) NO
         #tokenFeatures.append("lengthNext="+str(len(str(tNext)))) NO
         tokenFeatures.append("suf3Next="+tNext[-3:])
         tokenFeatures.append("suf5Next="+tNext[-5:])
         tokenFeatures.append("pre3Next="+tNext[:3])
         tokenFeatures.append("pre5Next="+tNext[:5])
         #tokenFeatures.append("postagNext="+pos_tags[k+1][1]) NO

         ###########Features not included in previous token##############
         '''if tNext.isupper():
            tokenFeatures.append("capNext=All")
         elif tNext[0].isupper():
            tokenFeatures.append("capNext=Ini")
         else:  
            tokenFeatures.append("capNext=None")'''
         '''if re.search(r'[0-9-]', tNext):
            if re.search(r'[a-zA-Z]', tNext):
               tokenFeatures.append("has_spcarN=Some")
            else:
               tokenFeatures.append("has_spcarN=All")
         else:
            tokenFeatures.append("has_spcarN=None")'''
         '''if tNext in ['Acid','acid'] or t in ['Acid','acid']:
            tokenFeatures.append("isAcid=Yes")
         else:
            tokenFeatures.append("isAcid=No")'''
      else:
         tokenFeatures.append("EoS")

      ###############################2-next token#################################
      '''if k<len(tokens)-2 and k != len(tokens)-1:
         tNext = tokens[k+2][0]
         tokenFeatures.append("formNext2="+tNext)
         tokenFeatures.append("suf3Next2="+tNext[-3:])
         tokenFeatures.append("pre3Next2="+tNext[:3])
      else:
         tokenFeatures.append("formNext2=None")'''
      
   
      result.append(tokenFeatures)
    
   return result

# Function to read the list of drugs, brands and groups and put them into separate lists
def get_resources(dir):
   drugs = []
   brands = []
   groups = []
   with open(dir, 'r') as file:
      # Read the file line by line
      for line in file:
         l = line.strip().split('|')
         if l[1] == 'drug':
            drugs.append(l[0].lower())
         elif l[1] == 'brand':
            brands.append(l[0].lower())
         elif l[1] == 'group':
            groups.append(l[0].lower())

   return drugs, brands, groups

# Function to read the list of drug names
def get_drug_names(dir):
   drug_names = []
   with open(dir, 'r') as file:
         # Read the file line by line
         for line in file:
            l = line.strip()
            drug_names.append(l.lower())
   return drug_names



## --------- MAIN PROGRAM ----------- 
## --
## -- Usage:  baseline-NER.py target-dir
## --
## -- Extracts Drug NE from all XML files in target-dir, and writes
## -- them in the output format requested by the evalution programs.
## --


# directories with files to process
datadir = sys.argv[1]
drugs_dir = sys.argv[2]
drug_names_dir = sys.argv[3]

# process each file in directory
for f in listdir(datadir) :
   
   # parse XML file, obtaining a DOM tree
   tree = parse(datadir+"/"+f)
   
   # process each sentence in the file
   sentences = tree.getElementsByTagName("sentence")
   for s in sentences :
      sid = s.attributes["id"].value   # get sentence id
      spans = []
      stext = s.attributes["text"].value   # get sentence text
      entities = s.getElementsByTagName("entity")
      for e in entities :
         # for discontinuous entities, we only get the first span
         # (will not work, but there are few of them)
         (start,end) = e.attributes["charOffset"].value.split(";")[0].split("-")
         typ =  e.attributes["type"].value
         spans.append((int(start),int(end),typ))
         

      # convert the sentence to a list of tokens
      tokens = tokenize(stext)
      # get drugs from resources
      drugs, brands, groups = get_resources(drugs_dir)
      # get drugs names from resources
      drug_names = get_drug_names(drug_names_dir)
      # extract sentence features
      features = extract_features(tokens, drugs, brands, groups, drug_names)

      # print features in format expected by crfsuite trainer
      for i in range (0,len(tokens)) :
         # see if the token is part of an entity
         tag = get_tag(tokens[i], spans) 
         print (sid, tokens[i][0], tokens[i][1], tokens[i][2], tag, "\t".join(features[i]), sep='\t')

      # blank line to separate sentences
      print()
