from transformers import pipeline
question_answerer = pipeline("question-answering")
print(question_answerer(
    question="Where do I work?",
    context="My name is Amir and I work at CL in in District of Columbia office."))
print(question_answerer(
    #question="Where is the capital of France?",
    #question="What is population Paris?",
    question="What is GDP of Paris?",
    context="""
    Paris (French pronunciation: (About this soundlisten)) is the capital and most 
    populous city of France, with an estimated population of 2,175,601 residents as
    of 2018,in an area of more than 105 square kilometres (41 square miles).[4] Since
    the 17th century, Paris has been one of Europe's major centres of finance, 
    diplomacy, commerce, fashion, gastronomy, science, and arts. The City of Paris is 
    the centre and seat of government of the region and province of le-de-France, or
    Paris Region, which has an estimated population of 12,174,880, or about 18 percent
    of the population of France as of 2017.[5] The Paris Region had a GDP of 709 billion 
    (808 billion)in 2017.[6] According to the Economist Intelligence Unit Worldwide Cost 
    of Living Survey in 2018, Paris was the second most expensive city in the world,
     after Singapore and ahead of Zurich, Hong Kong, Oslo, and Geneva.[7]"""
))