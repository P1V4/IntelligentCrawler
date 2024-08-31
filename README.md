# AUTOMATIC WEB SCRAPER

The purpose of the Automated Web Scraper is to extract the relevant attributes from web pages, without any hard-coding of the information to be scraped.

We plan to utilize the Large Language Models as well as their ability to generate responses based on queries, as well as their ability for code interpretation and generation.

A number of prompts for both the open-source and commercial models were engineered to extract attributes from different types of web pages.

The scrapers can be classified on the basis of the task that needs to be performed.

 

![Untitled](Untitled.png)

- Text-Based Extraction
    
    This approach directly feeds the concatenated text data that is visible on the webpage to an LLM and queries to extract relevant attributes.
    
    Sample Inputs:
    
    > Single Type Sites with Non-Explicit Ticker:
    
    	Input:
    		The Corbett Road Tactical Opportunity ETF takes a tactical investment     approach that adapts to changing market environment.
    		Performance as of 23/06/2023
    		CUS42
    		Net Assets(in dollars) 240,580,486
    		NAV(in $) 34.1 Daily change 0.23%
    	Output:
    		<json>{"ticker": "CUS42", "nav": "$34.1", "tna": "$240,580,486"}</json>
    
    Multiple Type Sites in Table-Like Format:
    	Input:
    		Ticker NAV(in $) Net Assets(millions)
    		RIS3 34.53 67.46
    		CAND 11.24 33.98
    		ZEQ 45.34
    	Output:
    		{"ticker":"RIS3","nav": "$34.53","tna": "$67,460,000"}
    		{"ticker":"CAND","nav": "$11.24","tna": "$33,980,000"}
    		{"ticker":"ZEQ","nav": "$45.34","tna": "null"}
    > 
    
    - Single
        - FLAN-T5-XXL - OpenSource
            - Prompt Design
                - v1
                    
                    Zero-shot approach
                    
                    ![Untitled](Untitled%201.png)
                    
                    - Very poor performance, didn’t have any understanding of the attributes and structure of response
                - v2
                    
                    Instruction with Question.
                    
                    ![Untitled](Untitled%202.png)
                    
                - v3
                    
                    Appending dynamic text on a common Passage and answering questions based on it.
                    
                    ![Untitled](Untitled%203.png)
                    
                    - This prompt is designed with dynamic appending and changes for every site.
                - v4
                    
                    Giving a fixed passage where the examples are out of the context..
                    
                    ```python
                    def t5_prompt(text_piece):
                    	task=f"""
                    	Task: Extract the relevant information from the given comprehension.
                    	Context: Read the following passage about information on assets and answer the questions based on the information provided. 
                      If you cannot find the answer to a question, provide "NA" as the answer.
                    
                    	"""
                    	infiltrate = """Decimal Point Analytics 
                    	Location Mumbai
                    	No. of Employees 368
                    
                    	Building Infrastructure
                    	Height
                    	Length
                    	Width
                    	20.3 cm
                    	34.3 cm
                    	123.24 cm"""
                    
                    	examples="""
                    
                    	Question and Example:
                    
                    	Input: Based on the passage, extract the values of Company Name, where Company name can also be referred as the name of the Building.
                    	Output: {'Company Name': 'Decimal Point Analytics'}
                    
                    	Input: Extract the values of Employee Strength and Building height from the above passage, where Employee Strength can also be referred as number of employees.
                    	Output: {'Employee Strength': '368 employees', 'Building Height': '20.3 cm'}
                    
                    	Input: Extract the values of the Company Name, Building Weight and Location from the above passage, where Location is the building location of the company.
                    	Output: {'Company Name': 'Decimal Point Analytics', Building Weight': "NA", 'Location': 'Mumbai'}
                    
                    	Instruction: Extract the values of Ticker, TNA and NAV from the above context, where Ticker is the Fund Code, TNA can also be referred as Total Net Assets and NAV can also be referred as current price.
                    	Output: {
                    	"""
                    	return task + f"{infiltrate}\n{text_piece}" +examples
                    ```
                    
                    - Poor Performance
                    - Didn’t understand the meaning of attributes properly
                    - The attributes extracted are different in examples than those required.
                - v5
                    
                    Included in context examples and structuring output.
                    
                    ![Untitled](Untitled%204.png)
                    
                    - Better understanding of the attributes and the output structure.
                    - Gave false positives from the values that are present in the example.
                - v6
                    
                    Examples in context as well as used delimiters and structure that was used in fine-tuning the model. 
                    
                    - Introduced examples covering individual recognition of every attribute in a disjunctive manner.
                    - Results are next to perfect and structured for ideal cases.
                    - False positives and small context length are a major drawback.
                    
                    ```python
                    def prompt_seq_gen(text_piece):
                    	
                    	examples="""
                    Context: Following Passages are about the details about the funds and their current market price and assets. 
                    Instruction: Extract the values of Ticker, NAV(Current Price) and TNA(Total Net Assets) from each of the following passages in json structured format. Ticker is the fund code. TNA can also be referred as Total Net Assets.
                    NAV can also be referred as the current market price per share. Do not add any other attributes that are not asked above. In case you don't find the value of the attributes asked, anser as "null".
                    Q: The Corbett Road Tactical Opportunity ETF takes a tactical investment approach that adapts to changing market environment.
                    Ticker CUS42
                    Net Assets(in dollars) 240,580,486
                    NAV(in $) 34.1 as of today
                    A: <json>{"ticker": "CUS42", "nav": "$34.1", "tna": "$240,580,486"}</json>
                    ###
                    Q: EmbossinG FUNDs Ticker:CRYI
                    A: <json>{"ticker": "CRYI", "nav":"null", "tna":"null"}</json>
                    ###
                    Q: NAV(in Dollars) 56.48 -0.03 TNA $365,454,474
                    A: <json>{"ticker":"null","nav": "$56.48", "tna":"$365,454,474"}</json>
                    ###
                    Q: Total Net Assets(millions) $32.460 as of 23/06/23
                    A: <json>{"ticker":"null","nav": "null","tna": "$32,460,000"}</json>
                    ###
                    Q: 
                    	"""
                    
                    	return examples+text_piece+"\nA:"
                    ```
                    
                - v7
                    
                    Expansion to more attributes with slight changes in the promp
                    
                    ```python
                    def prompt_seq_gen_2(text_piece):
                    	
                    	examples="""
                    Context: Following Passages are about the details about the funds and their current market price and assets. 
                    Instruction: Extract the values of Ticker, NAV(Value per Share), Inception Date(Current Date of Data) and TNA(Total Net Assets) from each of the following passages in json structured format. Ticker is the fund code. TNA can also be referred as Total Net Assets.
                    NAV can also be referred as the current market price per share. Do not add any other attributes that are not asked above. In case you don't find the value of the attributes asked, anser as "null".
                    Q: The Corbett Road Tactical Opportunity ETF takes a tactical investment approach that adapts to changing market environment.
                    Performance as of 23/06/2023
                    CUS42
                    Net Assets(in dollars) 240,580,486
                    NAV(in $) 34.1 Daily change 0.23%
                    A: <json>{"ticker": "CUS42", "nav": "$34.1", "tna": "$240,580,486", "inception date":"23/06/2023"}</json>
                    ###
                    Q: EmbossinG FUNDs Ticker:CRYI
                    A: <json>{"ticker": "CRYI", "nav":"null", "tna":"null", "inception date":"null"}</json>
                    ###
                    Q: NAV(in Dollars) 56.48 -0.03 TNA $365,454,474
                    A: <json>{"ticker":"null","nav": "$56.48", "tna":"$365,454,474", "inception date":"null"}</json>
                    ###
                    Q: Total Net Assets(millions) $32.460 as of May 7, 2023
                    A: <json>{"ticker":"null","nav": "null","tna": "$32,460,000", "inception date":"May 7, 2023"}</json>
                    ###
                    Q: 
                    	"""
                    
                    	return examples+text_piece+"\nA:"
                    ```
                    
            - Status:
                - Able to extract Relevant attributes. Confuses in units and symbols. Sometimes Ticker is difficult to extract if non-explicitly mentioned.
                - False-Positive is a major drawback. Can be omitted if TICKER values are a pre-requisite and the outputs can be filtered accordingly.
                - Some extra attributes are sometimes present, but we can omit them, because the response is in the form of structured JSON, with key-value pairs containing useful information.
            - Results on Large Database (42 Websites)
                - It worked better than KOR using GPT-3.5-turbo
                - Problems in areas where the relevant attributes are far from each other and appear in different subtrees. Therefore, as a result they may or may not get extracted. Also, if we assume to query the outputs by using the value of ticker as an input, it becomes infeasible to capture attributes that are not part of the same subtree.
                - In most cases the results were ideal.
                - Couldn’t recognize different keywords used for funds other than that given in the prompts. Eg. Couldn’t identify Total AUM as an identifier for Net Assets
                - Obtained a score of 29/42
            - Extension to include new attributes
        - KOR (GPT-3.5-turbo)
            - Prompt
                
                ```python
                schema = Object(
                	id="asset_info",
                	description = "Information or details about the asset or fund.",
                	attributes=[
                		Text(
                				id="ticker",
                				description="Ticker of the Asset",
                				examples=[("Ticker CRYI", "CRYI")],
                		),
                		Text(
                				id = "nav",
                				description = "The net asset value(NAV) of the fund or share. Can also be referred as Market Price or Current Price.",
                				examples=[("NAV(in Dollars) 56.48","$56.48")],
                		), 
                		Text(
                				id="tna",
                				description="Total Net Assets (TNA) of the share or fund. Can also be referred as Total Funds or Total Assets",
                				examples=[("Total Net Assets $2,460,580", "$2,460,580")],
                		),
                	],
                	examples =[(
                			"""
                The Corbett Road Tactical Opportunity ETF takes a tactical investment approach that adapts to changing market environment.
                Ticker	OPPX
                Net Assets	19,307,145
                –	–
                # of Holdings	28
                Fund Inception Date	02/25/2021
                Shares Outstanding	850,000
                
                NET ASSET VALUE (NAV)
                NAV	22.71
                Daily Change ($)	0.06
                Daily Change (%)	0.26%
                MARKET PRICE
                Closing Price	22.7190
                Daily Change ($)	0.0762
                """,
                				[
                					{"ticker":"OPPX", "nav":"$22.71", "tna": "$19,307,145"}
                				]
                				)
                	],
                	many= False,
                )
                ```
                
            - Status:
                - Is able to extract the terms effectively. With hardly a single false positive.
                - Is a scalable approach. Can be used for other attributes.
                - Obeys the instructions effectively.
            - Results on Large Database (42 New Websites)
                - Worked perfectly in ideal cases.
                - Less False Positive
                - Gave the values of attributes used in prompt sometimes instead of null values.
                - Long Context is confusing the model. Working better with a shorter context length.
        - KOR (GPT-4)
            - Perfect Extraction or Ideal Extraction is possible.
        - Generative Sequence Generation Models (GPT-NEO-X-20B)
            - Status
                - They are ineffective in extraction.
                - They try to complete the given context and generate new values that are not present in the given passage.
                - They confuse the context with the given few-shot examples.
                - Did not give feasible results for at least 2-3 sites.
    - Multiple
        - FLAN
            - Prompt:
                
                ```python
                def prompt_seq_gen_2(text_piece):
                	
                	examples="""
                Context: Following Passages are about the details about the funds and their current market price and assets. 
                Instruction: Extract the values of Ticker, NAV(Value per Share), Date of NAV(Date of NAV data) and TNA(Total Net Assets) from each of the following passages in json structured format. Ticker is the fund code. TNA can also be referred as Total Net Assets.
                NAV can also be referred as the current market price per share. Do not add any other attributes that are not asked above. In case you don't find the value of the attributes asked, anser as "null".
                Q: The Corbett Road Tactical Opportunity ETF takes a tactical investment approach that adapts to changing market environment.
                Performance as of 23/06/2023
                CUS42
                Net Assets(in dollars) 240,580,486
                NAV(in $) 34.1 Daily change 0.23%
                A: <json>{"ticker": "CUS42", "nav": "$34.1", "tna": "$240,580,486", "nav date":"23/06/2023"}</json>
                ###
                Q: EmbossinG FUNDs Ticker:CRYI
                A: <json>{"ticker": "CRYI", "nav":"null", "tna":"null", "nav date":"null"}</json>
                ###
                Q: NAV(in Dollars) 56.48 -0.03 TNA $365,454,474
                A: <json>{"ticker":"null","nav": "$56.48", "tna":"$365,454,474", "nav date":"null"}</json>
                ###
                Q: Total Net Assets(millions) $32.460 as of May 7, 2023
                A: <json>{"ticker":"null","nav": "null","tna": "$32,460,000", "nav date":"May 7, 2023"}</json>
                ###
                Q: 
                	"""
                
                	return examples+text_piece+"\nA:"
                ```
                
            - Status:
                - Is able to detect repetitive schema, and generate results.
                - Is able to extract data where sites have about 2-4 funds.
                - The prompt length is longer. Therefore the context length is limited.
                - For sites having more number of funds (10+), It mixes attributes of one fund with another, because of low context length about (300-400) which is less for sites having multiple funds.
        - KOR (GPT-3.5)
            - Prompt:
                
                ```python
                schema = Object(
                	id="asset_info",
                	description = "Information or details about the asset or fund.",
                	attributes=[
                		Text(
                				id="ticker",
                				description="Ticker of the Asset",
                				examples=[("Ticker CRYI", "CRYI")],
                		),
                		Text(
                				id = "nav",
                				description = "The net asset value(NAV) of the fund or share. Can also be referred as Market Price or Current Price.",
                				examples=[("NAV(in Dollars) 56.48","$56.48")],
                		), 
                		Text(
                				id="tna",
                				description="Total Net Assets (TNA) of the share or fund. Can also be referred as Total Funds or Total Assets",
                				examples=[("Total Net Assets $2,460,580", "$2,460,580")],
                		),
                	],
                	examples =[(
                			"""
                The Corbett Road Tactical Opportunity ETF takes a tactical investment approach that adapts to changing market environment.
                Ticker
                Net Assets(in dollars)
                NAV(in $)
                CUS42
                240,580,486
                34.1
                XPL
                324,752,763
                11.32
                """,
                				[
                					{"ticker":"CUS42", "nav":"$34.1", "tna": "$240,580,486"},
                					{"ticker":"XPL", "nav":"$11.32", "tna": "$324,752,763"},
                				]
                				)
                	],
                	many= True,
                )
                ```
                
            - Status:
                - Is successful in extracting multiple funds in majority of cases.
                - High context length, and a better understanding of schema and attributes because of a well defined prompt.
        - KOR (GPT-4)
            - Gives correct output in most of the cases.
            - Although, edge cases are missed sometimes.
- HTML-Code Based Extraction
    
    This type of approach uses the HTML code of the webpage as an input for the language model. Based on the HTML structure, the model is queried to extract the tags containing relevant information about the attributes. A suitable python code containing either the X-paths, Soup Tags or a complete extraction program is output from the model.
    
    This approach is effective in cases of multiple list-like or table-like structures where passing of the complete webpage will result into a large number of calls or would result in breaking of context to loose schematic information.
    
    > It was found that it is necessary for the parsing of HTML in the prettified form, for any type of model (even gpt-4) to provide accurate results.
    > 
    
    ```python
    html_string = '''
    <input><label >Your Password</label><div >Hello</div><div >|<a >Forgot Password?</a><a >Don\'t have an account yet?</a></div></div><div ><input ></input><button >Login</button></div></div></div></div></form><form id=\'buyerRegisterForm\'><div id=\'buyerRegister\'><div ><div ><div ><button ><span >×</span></button><h4 id=\'buyerRegisterLabel\'>Buyer Registration</h4></div><div ><div ><div ><span ><i ></i></span><span ><input id=\'buyerRegisterFirstName\'></input><label >Your First Name</label></span></div></div><div ><div ><span ><i ></i></span><span ><input id=\'buyerRegisterLastName\'></input><label >Your Last Name</label></span></div></div><div ><div ><span ><i ></i></span><span ><input id=\'buyerRegisterEmail\'></input><label >Your Email</label></span></div></div><div ><div ><span ><i ></i></span><span ><input id=\'password\'></input><label >Your Password</label></span></div></div><div ><div ><span ><i ></i></span><span ><input id=\'confirm_password\'></input><label >Confirm Password</label></span></div></div><div ></div><div ><a >Already have an account?</a></div></div><div ><input ></input><button >Register</button></div></div></div></div></form></div><noscript ><img ></img></noscript><noscript ><img ></img></noscript><div ></div><div ><noscript ></noscript></div><div ></div><div ></div><div ></div><div ></div></body></[document]>'
    '''
    
    html_string_prettified='''
    <input>
    	<label>Your Password</label>
    	<div>Hello</div>
    	<div>
    		<a>Forgot Password?</a>
    	</div>
    ....
    </input>
    '''
    ```
    
    - Approaches
        - Code Generation for Direct Attribute Output for Single Fund Information Sites
            
            This approach prompts the model to output a python program for extracting the relevant attributes.
            
            ```python
            def gpt_prompt(input_text):
            	prompt_gpt = """
            You are developing a code generation model to automate the process of creating Python code for web scraping. The model takes HTML code as input and generates Python code that can be used to scrape data from the provided HTML structure.
            -----
            Your task is to generate Python code based on the given HTML code string in python, which will enable the user to extract specific data elements from web pages. The Python code should use the BeautifulSoup library for parsing HTML and provide a flexible and scalable solution.
            -----
            Input:
            '''Extract the values of product name and product price from the below html code string.'''
            html = ""<html><head><title>Web Scraping Example</title></head><body><h1>Product List</h1><ul id="product-list"><li class="product"><div class="product-name">Product A</div><div class="product-price">$10.99</div></li><li class="product"><div class="product-name">Product B</div><div class="product-price">$19.99</div></li><li class="product"><div class="product-name">Product C</div><div class="product-price">$7.99</div></li></ul></body></html>""
            Output:
            ```
            from bs4 import BeautifulSoup
            
            soup = BeautifulSoup(html, 'html.parser')
            product_list = soup.find('ul', {'id': 'product-list'})
            products = product_list.find_all('li', {'class': 'product'})
            
            for product in products:
                name = product.find('div', {'class': 'product-name'}).text
                price = product.find('div', {'class': 'product-price'}).text
                print(f"Product Name: {name}")
                print(f"Product Price: {price}")
                print()
            ```
            	"""
            	return f"""
            {prompt_gpt}
            Input:
            '''Extract the values of TNA and NAV from the below html code string.''' 
            html = '''{input_text}'''
            Output: 
            ```
            	"""
            ```
            
        - Code Generation for Identifying Xpaths
            
            This approach prompts the model to generate the X-Paths of the relevant attributes. Further these XPaths are fed into a suitable python program which is executed to scrape the relevant attributes.
            
            ```python
            def prompt_gpt_3(input_text):
            
                instruction=f"""
            
            You are tasked with designing a portion of code for the Python web scraper that extracts specific attributes from a fund marketplace website from its HTML code.
            The attributes you need to extract are Ticker(Asset or Fund Code), NAV(Current Market Price), and TNA (Total Net Assets) using XPaths and prints them.
            
            Your task is to generate the XPaths for the required attributes as python strings which can be plugged in the scraper code. If any particular attribute is missing, generate a None python string for the respective attribute only.
            Examples:
            
            HTML Text Input:
            html = '''
            <html>
              <body>
                <div id="fund-details">
                  <div id="fund-info">
                    <h2 id="fund-name">Fund A</h2>
                    <div id="fund-holdings">
                      <ul>
                        <li>Holdings: A, B, C</li>
                        <li>Price Change: +10%</li>
                      </ul>
                    </div>
                  </div>
                  <div id="fund-data">
                    <div id="fund-code">Ticker:ABC123</div>
                    <div id="fund-nav">Price<span id='span_1'>$100.50</span></div>
                    <div id="div_3">$1,000,000</div>
                  </div>
                </div>
              </body>
            </html>
            
            '''
            Code:
            ```
            
            ticker_xpath = "//*[@id='fund_code']"
            nav_xpath = "//*[@id='span_1']"
            tna_xpath = "//*[@id='div_3']"
            ```
            
            HTML Text Input:
            html = '''
            <html>
              <body>
                <div>
                  <h2>Fund D</h2>
                  <div>
                    <ul>
                      <li>Holdings: P, Q, R</li>
                      <li>Price Change: +1%</li>
                    </ul>
                  </div>
                </div>
                <div>
                  <div>
                    <div id='div_0'>Ticker: <div id="div_5">MNO123</div></div>
                    <div id='div_1'>Price: <div id="div_3"><span>$</span>95.75</div></div>
                    <div id='div_2'>Net Assets: <div id="div_7">$3,500,000</div></div>
                  </div>
                </div>
              </body>
            </html>
            
            '''
            Code:
            ```
            ticker_xpath = "//*[@id='div_5']"
            nav_xpath = "//*[@id='div_3']"
            tna_xpath = "//*[@id='div_7']"
            ```
            
            HTML Text Input:
            html = '''
            <html>
              <body>
                <table class="fund-table">
                  <thead>
                    <tr>
                      <th id="t_0">Fund</th>
                      <th id="t_1">Ticker</th>
                      <th id="t_2">NAV</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td id="t_3">Fund A</td>
                      <td>ABC123</td>
                      <td id="t_6">$100.50</td>
                    </tr>
                  </tbody>
                </table>
              </body>
            </html>
            '''
            
            Code:
            ```
            ticker_xpath = "/html/body/table/tbody/tr/td[2]"
            nav_xpath = "//*[@id='t_6']"
            ## Since there is NO TNA, present in the html_code, Assigning None to tna_xpath
            tna_xpath = None
            ```
            Please generate the python strings containing the XPaths of the required attributes based on the structure and class names used in the target website. 
            
            Target Website:
            HTML Text Input:
            html='''
            {input_text}
            '''
            
            Code:
            ```
            """
                return instruction
            
            ```
            
            ```python
            def gen_program_3(html_input, model_output, path):
            
                ind2=model_output.find("```")
                if ind2 == -1:
                    code_text = model_output
                else:
                    code_text = model_output[:ind2]
                # print(code_text)
                model_output_trimmed = code_text
            
                program = f"""
            
            import lxml.html
            
            # Parse the HTML string
            tree = lxml.html.fromstring('''{html_input}''')
            
            # Define the XPaths for the required attributes
            {model_output_trimmed}
            
            # Extract the attribute values using XPaths
            
            res = ""
            # Print the extracted attribute values
            if ticker_xpath:
                try:
                    ticker_elements = tree.xpath(ticker_xpath)
                    text=''.join(ticker_elements[0].itertext())
                    res+="Ticker:"+ text.strip()+";"
                except:
                    res+="Ticker:"+ "Invalid XPaths"+";"
            else:
                res+="Ticker null"+";"
            
            if nav_xpath:
                try:
                    nav_elements = tree.xpath(nav_xpath)
                    text=''.join(nav_elements[0].itertext())
                    res+="Nav:"+ text.strip()+";"
                except:
                    res+="NAV:"+ "Invalid XPaths"+";"
            else:
                res+="Nav: null"+";"
            
            if tna_xpath:
                try:
                    tna_elements = tree.xpath(tna_xpath)
                    # print(tna_elements)
                    text=''.join(tna_elements[0].itertext())
                    res+="TNA:"+ text.strip()+";"
                except:
                    res+="TNA:"+ "Invalid XPaths"+";"
            else:
                res+="TNA: null"+";"
            # print(res)
            
            """
                return program
            ```
            
        - Code Generation for Identifying Soup Elements
            
            This approach prompts the model to output the relevant table soup element which is fed into a python program to extract the data of the table and save it as a csv file.
            
            ```python
            def table_element_prompt(input_html):
              table_gpt = f"""
            You are working on a project that involves extracting specific information from HTML documents using Beautiful Soup in Python. Your task is to generate Python code that, given an HTML document as input, extracts the Beautiful Soup object representing the table tag containing information about funds.
            
            Consider the following examples, each containing HTML code with multiple tables:
            Your task is to generate Python code that, when executed, extracts the Beautiful Soup object representing the table that contains information about funds and assign it to the 'funds_table' python variable. The code should be able to handle multiple tables and extract the correct one based on the table headers or any other suitable criteria.
            Use the 'id' attribute of the table tag to retrive the Beautiful Soup Object
            Examples:
            
            HTML Input:
            html = '''
            <!DOCTYPE html>
            <html>
            <head>
                <title>Financial Data</title>
            </head>
            <body>
                
                <table>
                    <thead>
                        <tr>
                            <th>Company</th>
                            <th>Revenue</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>Company A</td>
                            <td>1,000,000</td>
                        </tr>
                        <tr>
                            <td>Company B</td>
                            <td>2,500,000</td>
                        </tr>
                    </tbody>
                </table>
                <h1>Financial Data</h1>
                <table id="funds-table">
                    <thead>
                        <tr>
                            <th>TICKER</th>
                            <th>NAV</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>ABC003</td>
                            <td>$10.00</td>
                        </tr>
                        <tr>
                            <td>XYZ Fund</td>
                            <td>$25.00</td>
                        </tr>
                    </tbody>
                </table>
                <p>This is some additional text.</p>
                <div>
                    <table>
                        <thead>
                            <tr>
                                <th>Car Name</th>
                                <th>Mileage</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>Audi</td>
                                <td>500</td>
                            </tr>
                            <tr>
                                <td>BMW</td>
                                <td>1000</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </body>
            </html>
            '''
            Output:
            ```
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            tables = soup.find_all('table')
            
            ## Extracts and Saves the final Beautiful Soup Table object in the funds_table variable using id attribute, which contains the information about funds.
            funds_table = soup.find('table', id='funds-table')
            ```
            
            HTML Input:
            html = '''
            <!DOCTYPE html>
            <html>
            <head>
                <title>Financial Data</title>
            </head>
            <body>
                <table id="table_0">
                    <thead>
                        <tr>
                            <th>Company</th>
                            <th>Revenue</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>Company A</td>
                            <td>1,000,000</td>
                        </tr>
                        <tr>
                            <td>Company B</td>
                            <td>2,500,000</td>
                        </tr>
                    </tbody>
                </table>
                <table id="table_1">
                    <tr>
                        <td>
                            <table id="table_2">
                                <thead>
                                    <tr>
                                        <th>Fund Name</th>
                                        <th>Value</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <tr>
                                        <td>ABC Fund</td>
                                        <td>1000</td>
                                    </tr>
                                    <tr>
                                        <td>XYZ Fund</td>
                                        <td>2500</td>
                                    </tr>
                                </tbody>
                            </table>
                        </td>
                        <td>
                            <table id="table_3">
                                <thead>
                                    <tr>
                                        <th>Product</th>
                                        <th>Sales</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <tr>
                                        <td>Apple</td>
                                        <td>100</td>
                                    </tr>
                                    <tr>
                                        <td>Orange</td>
                                        <td>150</td>
                                    </tr>
                                    <tr>
                                        <td>Banana</td>
                                        <td>75</td>
                                    </tr>
                                </tbody>
                            </table>
            
                        </td>
                    </tr>
                </table>
                <table id="table_4">
                    <thead>
                        <tr>
                            <th>Car</th>
                            <th>Mileage</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>HONda</td>
                            <td>500</td>
                        </tr>
                        <tr>
                            <td>Hyundai</td>
                            <td>1000</td>
                        </tr>
                    </tbody>
                </table>
            </body>
            </html>
            '''
            Output:
            ```
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            tables = soup.find_all('table')
            
            ## Extracts and Saves the final Beautiful Soup Table object in the funds_table variable using id attribute, which contains the information about funds.
            ## Also extracts the lowest child table containing information about the funds to reduce unecessary information from other tables
            funds_table = soup.find('table', id='table_2')
            ```
            
            Please generate the Python code that accomplishes the task described above for the given target HTML code.
            
            Target HTML Code:
            html = '''
            {input_html}
            '''
            Output:
            ```
            """
              return table_gpt
            
            def gen_program_table_element(html_input, model_output, SAVE_PATH):
            
                ind2=model_output.find("```")
                if ind2 == -1:
                    code_text = model_output
                else:
                    code_text = model_output[:ind2]
                # print(code_text)
                model_output_trimmed = code_text
                program = f"""
            html = '''{html_input}'''
            {code_text}
            import pandas as pd
            # Extract the table data and store it in lists
            data = []
            headers = []
            print("Reached!")
            for row in funds_table.find_all('tr'):
                row_data = []
                for cell in row.find_all(['th', 'td']):
                    row_data.append(cell.text.strip())
                if len(row_data) > 0:
                    if not headers:
                        headers = row_data
                    else:
                    	if len(headers)>0:
                    		for i in range(len(row_data), len(headers)):
                    			row_data.append("null")
                    	data.append(row_data)
               #      else:
               #      	if len(headers)>0:
               #      		for i in range(len(row_data), len(headers)):
               #      			row_data.append("null")
            			# data.append(row_data)
            
            # Create a pandas DataFrame using the extracted data
            df = pd.DataFrame(data, columns=headers)
            
            # Save the DataFrame as a CSV file
            df.to_csv('{SAVE_PATH}', index=False)
            """
                return program
            ```
            
        - Chain Of Thought Reasoning in few-shots
            
            It was found that, when giving few-shots in the prompt, if we include a chain of though reasoning, that is provide a step-by-step explaination of the decision we made as comments in the sample generated code. The model takes them more seriously and helps in more specific and robust scraping.
            
            ```python
            ## Extracts and Saves the final Beautiful Soup Table object in the funds_table variable using id attribute, which contains the information about funds.
            ## Also extracts the lowest child table containing information about the funds to reduce unecessary information from other tables
            funds_table = soup.find('table', id='table_2')
            ```
            
    - Attribute Extraction
        - Without ids
            - It generates incorrect X-Paths for cases where a direct identifier to the tag is not present.
            - It does identify the feature to extract, but confuses in obtaining a suitable identifier.
            - Tries to match the text present in the element, which gives a poor match accuracy, because of different formatting and whitespaces present in the text.
        - With IDs
            - More Accurate Location in the HTML DOM
            - Sometimes Extracts Other Symboled Tags that are part of information,
                
                <div id=”div_1”><span id=”span1”>$</span>34.67<div>
                
                But it extracts just the $ sign.
                
            - Token Intensive, Number of subtrees doubled.
        - GPT vs StarCoder
            - GPT is more robust and outputs correct Xpaths while StarCoder sometimes uses id’s that were not part of the given html code.
            - Both fails to extract ticker values when it is present in non-explicit text form.
            - Number of false positives are high for StarCoder.
        
    - Table Extraction
        - Without ids
            - Is able to extract tables, but gives incorrect outputs where suitable identifier is not present.
        - With IDs
            - Is able to extract every table containing fund information into csv file, if it is present in the table tag.
            - Fails, when information is in the form of <div> or some other tag than
        - Multiple Tables
            - Prompted model to provide a list of table soup elements containing the relevant fund information.
            - Used output of the model to save the respective tables in separate CSV files.
            - Successful in the extraction of sites containing large databases having huge number of rows. A portion or subset of original code can be parsed in the model to retrieve the complete database.
    - Multiple Attribute Extraction
        
        This approach prompts the model to generate a suitable iterative code which can be used to extract repetitive html-structures of multiple funds.
        
        The model is prompted on a subset-html structure to provide an iterative loop like program based on the limited context. The prompt is designed in a way to apply this program into the complete webpage.
        
        This would lead to a complete extraction of pages containing a large number of multiple fund information where splitting can result into loss of schema or edge cases. 
        
        eg. [https://www.guggenheiminvestments.com/uit/secondary-uits](https://www.guggenheiminvestments.com/uit/secondary-uits)
        
        The program is then run on the complete webpage, scraping large tables or lists in a single output. 
        
        Prompt:
        
        ```python
        def multi_code_prompt(text_input):
            prompt= f"""
        You are given an HTML structure representing a webpage on multiple funds. 
        Design a Python program that can extract key attributes such as Ticker(Asset or Fund Code), NAV(current or daily price), and TNA(Total or Net Assets) of multiple funds listed on the webpage. 
        The multiple funds can be present with repeating HTML sub-structures.
        
        Example 1:
        html = '''
        <div class="contacts">
          <h2>Funds Classification</h2>
          <ul>
            <li>
              <h3>Ticker:</h3>
              <p>ABCV34</p>
              <h3>NAV:</h3>
              <p>$13.54</p>
              <h3>Net Assets:</h3>
              <p>$1,344,325</p>
            </li>
            <li>
              <h3>Ticker:</h3>
              <p>ABDV55</p>
              <h3>NAV:</h3>
              <p>$16.54</p>
              <h3>Net Assets:</h3>
              <p>$5,354,748</p>
            </li>
          </ul>
        </div>
        '''
        Code:
        ```
        from bs4 import BeautifulSoup
        
        soup = BeautifulSoup(html, 'html.parser')
        funds = []
        
        fund_elements = soup.find('ul').find_all('li')
        for fund_item in fund_elements:
            try:
                fund = {{}}
                ##Use Relative Position Indexing to Match Attributes
                fund['ticker'] = fund_item.find_all('h3')[0].find_next_sibling('p').text
                fund['nav'] = fund_item.find_all('h3')[1].find_next_sibling('p').text
                fund['tna'] = fund_item.find_all('h3')[2].find_next_sibling('p').text
                funds.append(fund)
            except:
                pass
        ```
        Example2:
        html='''
        <table class="funds">
          <tr>
            <th>Fund Code</th>
            <th>Price($)</th>
            <th>Change</th>
            <th>TNA(in millions)</th>
          </tr>
          <tr>
            <td>RFGE</td>
            <td>34.66</td>
            <td>+0.23%</td>
            <td>345.21</td>
          </tr>
          <tr>
            <td>ABSH</td>
            <td>11.26</td>
            <td>+1.56%</td>
            <td>89.6</td>
          </tr>
        </table>
        '''
        Code:
        ```
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        funds = []
        
        fund_elements = soup.find('table', class_='funds')
        
        for element in fund_elements.find_all('tr')[1:]:
            fund = {{}}
            try:
                cells = element.find_all('td')
                ##Use Relative Position Indexing to Match Attributes
                fund['ticker'] = cells[0].text
                fund['nav'] = cells[1].text
                fund['tna'] = cells[3].text
                funds.append(fund)
            except:
                pass    
        ```
        Your task is to generate a python program for extracting the above mentioned attributes of multiple funds in the following target website.
        Target Website:
        html = '''
        {text_input}
        '''
        Code:
        ```
        """
            return prompt
        
        ```
        
- NER-Based Scraper
    - NER- Model
        - Using LLMs
            
            ```python
            def NER_PROMPT(input_text):
            	prompt=f"""
            	Prompt: "Classify the following word/sentence into one of the following classes: Ticker, NAV, TNA or others."
            
            	Examples:
            
            	1. Input: "AAPL"
            	   Output: Class1 (Ticker)
            
            	2. Input: "$50."
            	   Output: Class2 (NAV)
            
            	3. Input: "$1 million <TICKER>."
            	   Output: Class3 (TNA)
            
            	4. Input: "What is the current NAV and TNA of the ABC mutual fund?"
            	   Output: Class4 (others)
            
            	5. Input: "Ticker: XYZ."
            	   Output: Class1 (Ticker)
            
            	6. Input: "$10,9334,543."
            	   Output: Class3 (TNA)
            
            	7. Input: "NAV: $33.02."
            	   Output: Class2 (NAV)
            
            	8. Input: "The ticker is used to EXTRACT."
            	   Output: Class4 (others)
            
            	9. Input: "Heef dd $3.54 billion."
            	   Output: Class3 (TNA)
            
            	10. Input: "NAV: $56.99"
            	    Output: Class2 (NAV)
            
            	11. Input: "Change in price: 3.45% "
            		Output: Class4 (others)
            	Task:
            	Classify the following sentence or word in one of the classes.
            	Input: "{input_text}"
            	Output: 
            	"""
            	return prompt
            ```
            
    - Algorithm
        
        ![Untitled](Untitled%205.png)
        
        ![Untitled](Untitled%206.png)
        
- Utilities
    - Subtree Generation
        
        ![Untitled](Untitled%207.png)
        
    - Dynamic Rendering of Javascript using Playwright
        
        ```python
        def HTMLFromURL(url: str, extra_sleep: int):
            """Download an HTML from a URL.
            
            In some pathological cases, an extra sleep period may be needed.
            """
            with sync_playwright() as p:
                browser = p.chromium.launch()
                page = browser.new_page()
                page.goto(url, wait_until="load")
                if extra_sleep:
                    sleep(extra_sleep)
                html_content = page.content()
                browser.close()
            return html_content
        ```
        
    - HTML Cleaning and Soup.Prettify for parsing HTML into Model.
- Summary
    
    
    - Single-Fund
        
        The scores are marked for 27 sites that were not seen during the engineering of prompt.
        
        Partial marking is given if some of the attributes are present in the output. 
        
        - GPT-4 doesn’t give any false positives and gives exact values for all the cases.
        - Flan has high false positives but the score is given assuming a Ticker Filter.
        
        |  | FLAN-T5-XXL | Star Coder | GPT-3.5-Turbo | GPT-4 |
        | --- | --- | --- | --- | --- |
        | Text | 18 | N/A | 16.5 | 27 |
        | Code | N/A | 11.7 | 13.8 | N/A |
    - Multiple-Funds
        - KOR-GPT-3.5 struggles to extract tables. out of 15 sites.
        
        |  | GPT-3.5-Turbo | GPT-4 |
        | --- | --- | --- |
        | Text | 3.5 | 12.3 |
        | Code | 8.8 | N/A |
- An end-to-end pipeline for scraper code generation about contacts of people
    - We extended our Mutiple Funds Extraction Approach using Iterative Code Generation Technique to create an end-to-end pipeline for scraping multiple contact information from websites
    - The model used was GPT-4.
    - A colony of scrapers on the same website were generated using different subtree context.
    - The best scraper among them was identified depending upon the quality of data that was generated. This process is subjective.
    - The pipeline takes a URL as an input. And generates its python scraper file along with the excel scraped data.
    - It was found that the best scraper is always found in the colony.
    
- Conclusion
    
    Single Fund Sites: GPT-4 with KOR is the perfect ideal option. After that, FLAN-T5-XXL also generates satisfactory results in majority of cases, if post-process filtering is possible.
    
    Multiple Fund Sites: GPT-4 with KOR is the ideal option. If the site is large or is in the form of tables, then GPT-3.5 with Iterative Multi Prompt Design should be chosen.
    
- Directory Structure
    - Code Scraper
        - Files
            - Outputs
                
                Contains Outputs for Different Experiments
                
            - Results
                
                Contains Results for the Final Pipelines Run on Different Datasets
                
        - autoTokenizer.py
            
            “” Encoder, Decoder and Tokenization Functions for Hugging Face Models. “”
            
        - code_prompt_generator and code_prompt_generator_2.py
            
            Contains different types of prompts used in the pipeline
            
        - code_scraper_pipe_hf.py
            
            Pipeline for code generation models using a Hugging Face Inference model. The URLS are taken from an excel file as in input, the corresponding outputs are saved in the output excel file. The prompt and the program generator functions can be chosen at the header variables along with other parameters.
            
        - code_scraper_pipeline.py
            
            Pipeline for code generation models using a openai GPT Inference model. The URLS are taken from an excel file as in input, the corresponding outputs are saved in the output excel file. The prompt and the program generator functions can be chosen at the header variables along with other parameters.
            
        - contacts_code_scraper_pipeline.py
            
            Similar as the above pipelines, just initialized with the peeople’s contacts scraping prompt.
            
        - dom_maker.py
            
            This file contains functions which create the DOM tree object out of the given HTML string.
            
        - gpt_code.py
            
            this contains functions for inferencing the GPT models from openai along with tokenization.
            
        - hf_models.py
            
            this contains functions for inferencing the Hugging Face models from openai along with tokenization.
            
        - hf_trial.py
            
            this contains functions for inferencing the Hugging Face models from openai along with tokenization.
            
        - html_extractor.py
            
            THis contains utility function for cleaning and appending ids to the HTML code. Also contains function for extracting HTML from local files.
            
        - multiple_code_prompt.py
            
            File containing prompt for scraping multiple fund database using code generation
            
        - prog_executor.py
            
            Contains function which executes the given python program string with handling exceptions.
            
        - subtree_generator.py
            
            Contains functions for splitting of HTML page into different subtrees.
            
        - t5_tokenizer.py
            
            this contains functions for inferencing the Flan-t5-models models along with tokenization.
            
        - treelib_visualize.py
            
            This contains functions for visualizing any tree using the treelib python library.
            
        - url_to_html.py
            
            This contains functions for dynamic rendering of JavaScript from the given URL and outputting its HTML code. 
            
    - WebScraperPipeline
        - Files
            - Outputs
                
                Contains Outputs for Different Experiments
                
            - Results
                
                Contains Results for the Final Pipelines Run on Different Datasets
                
        - autoTokenizer.py
            
            “” Encoder, Decoder and Tokenization Functions for Hugging Face Models. “”
            
        - prompt_generators.py
            
            Contains different types of prompts used in the pipeline
            
        - web_scraper.py
            
            Pipeline for text-based models using a prompt. The URLS are taken from an excel file as an input, the corresponding outputs are saved in the output excel file. The prompt and the model can be chosen at the header variables along with other parameters.
            
        - web_scraper_kor.py
            
            Pipeline for text-based models using a KOR. The URLS are taken from an excel file as in input, the corresponding outputs are saved in the output excel file. The KOR schema and the model can be chosen at the header variables along with other parameters.
            
        - dom_maker.py
            
            This file contains functions which create the DOM tree object out of the given HTML string.
            
        - gpt_models.py or gpt_trial.py
            
            this contains functions for inferencing the GPT models from openai along with tokenization.
            
        - hf_models.py
            
            this contains functions for inferencing the Hugging Face models from openai along with tokenization.
            
        - html_generator.py
            
            THis contains utility function for cleaning and appending ids to the HTML code. Also contains function for extracting HTML from local files.
            
        - kor_info_multiple.py
            
            This contains the KOR schema for scraping multiple contacts sites.
            
        - kor_multiple.py
            
            This contains the KOR schema for scraping multiple funds sites.
            
        - kor_trial.py
            
            This contains the KOR schema for scraping single funds sites.
            
        - prog_executor.py
            
            Contains function which executes the given python program string with handling exceptions.
            
        - subtree_generator.py
            
            Contains functions for splitting of HTML page into different subtrees.
            
        - t5_tokenizer.py
            
            this contains functions for inferencing the Flan-t5-models models along with tokenization.
            
        - text_splitter.py
            
            contains function for splitting the text into different subsets and adding prompts to each of them.
            
        - treelib_visualize.py
            
            This contains functions for visualizing any tree using the treelib python library.
            
        - url_to_html.py
            
            Contains function for dynamic loading of the webpage.
            
    - Final Results
        
        Contains Compilation of the Final Results for All the Approaches
        
    - Final_Code_Scraper
        
        An end-to-end pipeline program for generating scrapers for sites containing multiple contact information.
        
        Usage:
        
        Terminal>> python contacts_code_scraper_pipeline.py [URL]
        
        In the Outputs folder, the resulting scraped csv and the scraper code will be generated.
        
    - NER
        
        For Named Entity Recognition Model using LLMs
        
    - NER_Extractor
        
        For scraping information from sites containing recurring HTML structure using the NER based approach and LLM.
        
    - Web Scraper
        
        Some Initials Files and experiments before converting them into pipelines.
