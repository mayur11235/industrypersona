import os
from os.path import dirname, abspath
import openai
from dotenv import load_dotenv
import re

script_path = abspath(dirname(__file__))
load_dotenv()

class PromptGenerator:
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.environ['OPENAI_API_KEY'])
        self.roles = ['C-suite executive','CEO','CFO','COO','CTO', 'CIO', 'CMO', 'CHRO', 'CLO', 'CSO', 'CDO', 'CAIO']
        self.inds = ['Fortune 500',
                    'Consumer',
                    'Energy, Resources & Industrials',
                    'Financial Services',
                    'Government & Public Services',
                    'Life Sciences & Health Care',
                    'Technology, Media & Telecommunications']
        
    def get_response(self,messages):
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=1000
        )
        return response.choices[0].message.content
    
    def persona_prompt_template(self,role,ind):
        return [{"role": "system", "content":f"""Generate a prompt for a persona that is {role} of a {ind} company. The prompt should include the following:\
            Purpose, background, interests, values, communication, tone, style adaptations, backstory and personality. These should be in order and the output should be one per line. \
            For example, a prompt for a CFO of a Technology, Media & Telecommunications company should look like this:
            Purpose: The purpose of this CFO persona is to engage with stakeholders, including investors, employees, and partners, in discussions related to financial strategies, investments, and economic forecasts within the technology sector. This persona will play a crucial role in presenting financial insights, making strategic decisions, and communicating the financial health and outlook of the company. The target audience includes senior management, the board of directors, investors, and financial analysts. The desired outcome is to build trust, provide clarity on financial matters, and support strategic business decisions through effective communication and financial acumen.
            Background: Holds a degree in Finance or Accounting, with an MBA or similar advanced degree. Has over 15 years of experience in financial roles within the technology sector, including positions in financial analysis, budgeting, and strategic planning, culminating in the CFO role.
            Interests: Keen interest in emerging technologies, financial markets, and economic trends. Invests personal time in staying ahead of industry changes, regulatory developments, and innovation in financial management practices.
            Values: Transparency, integrity, and sustainability. Believes in making data-driven decisions, investing in innovation, and fostering a culture of financial responsibility and ethical business practices.
            Communication: Professional and articulate, yet accessible and engaging. Capable of breaking down complex financial concepts into understandable terms for non-financial audiences.
            Tone: Generally formal in professional settings, especially in written communications and public speaking engagements, but adopts a more conversational and supportive tone in one-on-one meetings or team discussions.
            Style Adaptations: Adjusts communication style based on the audience and context. For example, uses more technical language and details when discussing with finance professionals, while focusing on strategic implications and broader impacts in discussions with non-financial stakeholders.
            Backstory: This CFO began their career in a financial analyst role at a startup technology company, experiencing firsthand the challenges of managing finances in a high-growth, rapidly changing environment. Through a combination of strategic foresight, operational efficiency, and a commitment to innovation, they climbed the ranks to become CFO. Along the way, they played a pivotal role in navigating the company through financial downturns and fundraising rounds, leading to its current position as a leader in the tech industry. This journey has instilled a deep understanding of the importance of adaptability, strategic investment in technology, and the impact of financial leadership on company culture and success.
            Personality: Analytical, detail-oriented, forward-thinking, and strategic."""}]
    
    def update_all_txts(self):
        for r in self.roles:
            for i in self.inds:
                print(f'Processing {r} {i}')
                output = self.get_response(self.persona_prompt_template(r,i))
                output = re.sub(r'\n+', '\n', output)
                formatted_r = 'All' if r == 'C-suite executive' else r
                formatted_i = 'All' if i == 'Fortune 500' else i.split()[0].split(',')[0]

                persona_txt_file = os.path.join(script_path,"resources" ,f'{formatted_r}_{formatted_i}.txt')
                with open(persona_txt_file, 'w') as file:
                    file.write(output)

    def update_single_txt(self, r,i):
        print(f'Processing {r} {i}')
        output = self.get_response(self.persona_prompt_template(r,i))
        output = re.sub(r'\n+', '\n', output)
        formatted_r = 'All' if r == 'C-suite executive' else r
        formatted_i = 'All' if i == 'Fortune 500' else i.split()[0].split(',')[0]

        persona_txt_file = os.path.join(script_path,"resources","prompts",f'{formatted_r}_{formatted_i}.txt')
        with open(persona_txt_file, 'w') as file:
            file.write(output)

if __name__ == "__main__":
    pg = PromptGenerator()
    #pg.update_all_txts()
    pg.update_single_txt('CTO','Technology, Media & Telecommunications')
    