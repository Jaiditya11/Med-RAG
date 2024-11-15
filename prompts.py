context = """Purpose: The primary role of this agent is that it should be able to generate
a list of Diseases and their treatments in text."""

code_parser_template="""Parse the response from a previous LLM into a description and a string of valid code,also come up with a valid filename this
could be saved as that doesnt contain special characters.Here is the response {response}. You should pass this in the following JSON format"""