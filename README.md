## Description:
  
  Our project idea is to create a software that measures the accuracy of different audio transcription models based on what phonetics/words they are poor at accurately transcribing. This will be a multi part process of development. We will need to gather data that can be used to link words to different phonetics. We will then need to develop a method of reading passages into transcription models, in such a way that misses demonstrate a trend. We then need to develop our program so that it can check the words that a model misses, correlate them with phonetics, and then use data analysis to conclude what phonetics/words that model struggles with.
  
## Rough Timeline

### 🔵 Mid October
- Collect phonetic and word-level data and build a local database linking words to their corresponding phonetic components.
- Select or develop sufficiently large reading passages that contain a representative sample of each phonetic.
- Ensure the passages provide enough examples for transcription errors to reveal meaningful phonetic patterns.

### 🔵 Mid November
- Develop a standardized method for providing spoken versions of the selected passages to audio transcription models.
- Explore different input methods, such as prerecorded human speech or AI-generated speech.
- Establish a consistent testing pipeline so that transcription models can be evaluated under comparable conditions.

### 🔵 End of Semester
- Evaluate several selected transcription models using the standardized dataset and testing pipeline.
- Collect transcription-error data and analyze patterns using clustering and other appropriate methods.
- Identify broader trends in which phonetics are most difficult for transcription models to recognize accurately.
- Summarize the findings in a final report and presentation.

## Goals:
  
  Our goal is to create a software that measures what phonetics a particular transcription model struggles to transcribe accurately, and in measuring multiple models, further come to find a trend of what phonetics are difficult to transcribe on the whole.

## Data:

  - Collect phonetic and word libraries that link phonetics with words
  - Use SQL or other data management tools to create our own local database from pieces gathered online
  - Analyze trends based on transcription model testing
  - Resources:
    - Phonetics Dictionary: https://github.com/cmusphinx/cmudict/blob/master/cmudict.dict
    - American English Phonetic Inventory (All or most phonetics used in American English): https://phoible.org/inventories/view/2176 
