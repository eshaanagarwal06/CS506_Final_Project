# Description
  
  Our project idea is to create a software that measures the accuracy of different audio transcription models based on what phonetics/words they are poor at accurately transcribing. This will be a multi part process of development. We will need to gather data that can be used to link words to different phonetics. We will then need to develop a method of reading passages into transcription models, in such a way that misses demonstrate a trend. We then need to develop our program so that it can check the words that a model misses, correlate them with phonetics, and then use data analysis to conclude what phonetics/words that model struggles with.
  
  ## Rough Timeline
    
    ### Mid October
     - Successfully collected phonetic/word data and created a local database of our own that links words to different phonetics.
     - Selected/devised large texts that incorporate a fair sample size of each phonetic through the words used, such that miss translations on         the side of the transcription model can be interpreted as significant.
    
    ### Mid November
     - Developed method of feeding read text (either by AI, saved recordings of each reading the passages, etc.) to an audio transcription model        with clarity to effectively test model
    
    ### End of semester
     - Tested several select transcription models against significant data
       Collected data and analyzed it with clustering methods
     - Come to conclusions about general trend of difficult to transcript phonetics
     - Create final report/presentation

  ## Goals
  
  Our goal is to create a software that measures what phonetics a particular transcription model struggles to transcribe accurately, and in measuring multiple models, further come to find a trend of what phonetics are difficult to transcribe on the whole.

  ## Data
  - Collect phonetic and word libraries that link phonetics with words
  - Use SQL or other data management tools to create our own local database from pieces gathered online
  - Analyze trends based on transcription model testing
  - Resources:
    - Phonetics Dictionary: https://github.com/cmusphinx/cmudict/blob/master/cmudict.dict
    - American English Phonetic Inventory (All or most phonetics used in American English): https://phoible.org/inventories/view/2176 
