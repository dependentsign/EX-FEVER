import pandas as pd
import json
import random
import openai
import sqlite3
import time
from sklearn.metrics import confusion_matrix
import os 
import argparse

cursor = sqlite3.connect('data/wiki_db.db').cursor()

test_file = 'data/mini_test.csv'
test_file = pd.read_csv(test_file)


openai.api_key = 'your_api_key'

exp3_prompt = """
I will provide you with evidence and a claim. Your task is to determine if the claim is supported, refuted, or if there is not enough information based on the given evidence. You need to choose one of the following labels: 'SUPPORT', 'REFUTE', or 'NOT ENOUGH INFO'. After choosing a label, please provide a brief explanation for your choice.

Example 1:
Evidence: Alexander Rae Baldwin III (born April 3, 1958) is an American actor, film producer, comedian, and political activist. The oldest of the Baldwin brothers, he is known for his versatile performances, from comic work on television to dramatic roles in film. Baldwin has received various accolades, including three Emmy Awards, three Golden Globe Awards and eight Screen Actors Guild Awards, as well as nominations for an Academy Award, a Tony Award and a British Academy Film Award.
In his early career, Baldwin played both leading and supporting roles in a variety of films such as Tim Burton's Beetlejuice (1988), Mike Nichols' Working Girl (1988), Jonathan Demme's Married to the Mob (1988), and Oliver Stone's Talk Radio (1988). He gained attention for his performances as Jack Ryan in The Hunt for Red October (1990) and in Glengarry Glen Ross (1992). Since then he has worked with directors such as Woody Allen in Alice (1990), To Rome with Love (2012) and Blue Jasmine (2013), and Martin Scorsese in The Aviator (2004) and The Departed (2006). His performance in the drama The Cooler (2003) garnered him a nomination for the Academy Award for Best Supporting Actor. He has done voice work for The SpongeBob SquarePants Movie (2004), Madagascar: Escape 2 Africa (2008), Rise of the Guardians (2012), and The Boss Baby film franchise (2017–present).
From 2006 to 2013, Baldwin received critical acclaim starring alongside Tina Fey as Jack Donaghy on the NBC sitcom 30 Rock, winning two Primetime Emmy Awards, three Golden Globe Awards, and seven Screen Actors Guild Awards for his work on the series, making him the male performer with the most SAG Awards in history. On stage, he portrayed Stanley Kowalski in the 1992 Broadway production of A Streetcar Named Desire and the title character in a 1998 Off-Broadway production of Macbeth, the former earning him a Tony Award nomination. Baldwin co-starred in Mission: Impossible – Rogue Nation (2015) and Mission: Impossible – Fallout (2018), the fifth and sixth installments of the Mission: Impossible series. He is also a columnist for The Huffington Post. He also was the host of Match Game from 2016 until 2021.
Baldwin has received critical acclaim for his portrayal of Donald Trump on the long-running sketch series Saturday Night Live, both during the latter part of the 2016 presidential election campaign and following the inauguration, a role that won him his third Primetime Emmy in 2017. He was nominated again in 2018 and 2021.In 2021, while on the Rust film set, Baldwin discharged a revolver used as a prop, which killed cinematographer Halyna Hutchins and injured director Joel Souza.Married to the Mob is a 1988 American crime comedy film directed by Jonathan Demme, and starring Michelle Pfeiffer, Matthew Modine, Dean Stockwell, Mercedes Ruehl, and Alec Baldwin. Pfeiffer plays Angela de Marco, a gangster's widow from Brooklyn, opposite Modine as the undercover FBI agent assigned the task of investigating her mafia connections.
The film was released on August 19, 1988, by Orion Pictures. It earned positive reviews from critics and earned several accolades; Pfeiffer was nominated for a Golden Globe Award for Best Actress – Motion Picture Comedy or Musical, and Stockwell was nominated for an Academy Award for Best Supporting Actor.
Check the claim: 'Joseph Morgan is an American actor known for his role in the 1988 American crime comedy film directed by Jonathan Demme.' from the above evidence
Choices: ['SUPPORT', 'REFUTE', 'NOT ENOUGH INFO']
Answer: REFUTE
Explanation: Alec Baldwin, not Joseph Morgan, is an American actor known for his role in Jonathan Demme's Married to the Mob in 1988. Married to the Mob is a 1988 American crime comedy film directed by Jonathan Demme. 

Example 2:
Evidence: The Woman in Red is a 1984 American romantic comedy film directed by and starring Gene Wilder. Wilder also wrote the script, adapting it from the Yves Robert film Pardon Mon Affaire (Un éléphant ça trompe énormément). It co-stars Charles Grodin, Gilda Radner, Joseph Bologna, Judith Ivey, and Kelly LeBrock. The film won an Academy Award for Best Original Song for "I Just Called to Say I Love You", written and performed by Stevie Wonder.Kelly LeBrock (born March 24, 1960) is an American actress and model. Her acting debut was in The Woman in Red (1984), co-starring Gene Wilder. She also starred in the films Weird Science (1985), directed by John Hughes, and Hard to Kill (1990), with Steven Seagal.Weird Science is a 1985 American science fantasy comedy film written and directed by John Hughes and starring Anthony Michael Hall, Ilan Mitchell-Smith, and Kelly LeBrock. The title is taken from a pre-Comics Code Authority 1950s EC Comics magazine of the same name, the rights to which were acquired by the film's producer Joel Silver. The title song was written and performed by American new wave band Oingo Boingo.
Check the claim: 'The Woman in Red is a 1984 American romantic comedy film with a co-star, an American actress and model who also starred in a 1985 American science fantasy comedy film.' from the above evidence
Choices: ['SUPPORT', 'REFUTE', 'NOT ENOUGH INFO']
Answer: SUPPORT
Explanation: The Woman in Red is a 1984 American romantic comedy film with co-star Kelly LeBrock. Kelly LeBrock is an American actress and model who also starred in the film Weird Science. Weird Science is a 1985 American science fantasy comedy film. 

Example 3:

Evidence: Toy Story is a 1995 American computer-animated comedy film produced by Pixar Animation Studios and released by Walt Disney Pictures. The first installment in the  Toy Story franchise, it was the first entirely computer-animated feature film, as well as the first feature film from Pixar. The film was directed by John Lasseter (in his feature directorial debut) and written by Joss Whedon, Andrew Stanton, Joel Cohen, and Alec Sokolow from a story by Lasseter, Stanton, Pete Docter, and Joe Ranft. The film features music by Randy Newman, was produced by Bonnie Arnold and Ralph Guggenheim, and was executive-produced by Steve Jobs and Edwin Catmull. The film features the voices of Tom Hanks, Tim Allen, Don Rickles, Jim Varney, Wallace Shawn, John Ratzenberger, Annie Potts, R. Lee Ermey, John Morris, Laurie Metcalf, and Erik von Detten. Taking place in a world where toys come to life when humans are not present, the plot focuses on the relationship between an old-fashioned pull-string cowboy doll named Woody and a modern astronaut action figure, Buzz Lightyear, as they evolve from rivals competing for the affections of their owner, Andy Davis, to friends who work together to be reunited with Andy after being separated from him.
Following the success of their 1988 short film Tin Toy, Pixar was approached by Disney to produce a computer-animated feature film told from a small toy's perspective. Lasseter, Stanton, and Docter wrote early story treatments, which were rejected by Disney, who wanted the film's tone to be "edgier". After several disastrous story reels, production was halted and the script was rewritten to better reflect the tone and theme Pixar desired: "toys deeply want children to play with them, and ... this desire drives their hopes, fears, and actions". The studio, then consisting of a relatively small number of employees, produced the film under only minor financial constraints.
Toy Story premiered at the El Capitan Theatre in Los Angeles, California, on November 19, 1995, and was released in theaters in North America on November 22. It was the highest-grossing film during its opening weekend, eventually grossing over $373 million worldwide, making it the second highest-grossing film of 1995. The film received critical acclaim, and holds a 100% approval rating on Rotten Tomatoes. It was praised for the technical innovation of the 3D animation, wit and thematic sophistication of the screenplay, musical score, and vocal performances (particularly Hanks and Allen); it is considered by many to be one of the best animated films ever made. The film received three Academy Award nominations (Best Original Screenplay (the first animated film to be nominated for this award), Best Original Song for "You've Got a Friend in Me", and Best Original Score) as well as winning a Special Achievement Academy Award. In 2005, the film was selected for preservation in the United States National Film Registry by the Library of Congress as being "culturally, historically, or aesthetically significant", one of seven films designated in its first year of eligibility. The success of Toy Story launched a multimedia franchise and a series of three sequels, starting with Toy Story 2 (1999). The film also had a theatrical 3-D re-release in 2009 as a part of a double feature with the second film.
A spin-off, Lightyear, was released in 2022, with Chris Evans voicing the in-universe human Buzz Lightyear who inspired the action figure toyline in the Toy Story films.This list provides an overview of animated productions that can be considered as milestones in the development of animation techniques or in artistic or commercial success.
Check the claim: 'Toy Story was released by Walt Disney Home Video on VHS and Laser Disc in the United States on October 29, 1996.' from the above evidence
Choices: ['SUPPORT', 'REFUTE', 'NOT ENOUGH INFO']
Answer: NOT ENOUGH INFO
Explanation: No information shows that Toy Story was released by Walt Disney Home Video on VHS and Laser Disc in the United States on October 29, 1996. 

Now, please evaluate the following claim based on the provided evidence:

"""

exp1_prompt = """
I will provide you with evidence and a claim. Your task is to determine if the claim is supported, refuted, or if there is not enough information based on the given evidence. You need to choose one of the following labels: 'SUPPORT', 'REFUTE', or 'NOT ENOUGH INFO'. After choosing a label, please provide a brief explanation for your choice.

Example:

Evidence: The Woman in Red is a 1984 American romantic comedy film directed by and starring Gene Wilder. Wilder also wrote the script, adapting it from the Yves Robert film Pardon Mon Affaire (Un éléphant ça trompe énormément). It co-stars Charles Grodin, Gilda Radner, Joseph Bologna, Judith Ivey, and Kelly LeBrock. The film won an Academy Award for Best Original Song for "I Just Called to Say I Love You", written and performed by Stevie Wonder.Kelly LeBrock (born March 24, 1960) is an American actress and model. Her acting debut was in The Woman in Red (1984), co-starring Gene Wilder. She also starred in the films Weird Science (1985), directed by John Hughes, and Hard to Kill (1990), with Steven Seagal.Weird Science is a 1985 American science fantasy comedy film written and directed by John Hughes and starring Anthony Michael Hall, Ilan Mitchell-Smith, and Kelly LeBrock. The title is taken from a pre-Comics Code Authority 1950s EC Comics magazine of the same name, the rights to which were acquired by the film's producer Joel Silver. The title song was written and performed by American new wave band Oingo Boingo.
Check the claim: 'The Woman in Red is a 1984 American romantic comedy film with a co-star, an American actress and model who also starred in a 1985 American science fantasy comedy film.' from the above evidence
Choices: ['SUPPORT', 'REFUTE', 'NOT ENOUGH INFO']
Answer: SUPPORT
Explanation: The Woman in Red is a 1984 American romantic comedy film with co-star Kelly LeBrock. Kelly LeBrock is an American actress and model who also starred in the film Weird Science. Weird Science is a 1985 American science fantasy comedy film. 

Now, please evaluate the following claim based on the provided evidence:

"""


def main():
        
    succ = 0
    c = 0
    with open(f'results/mini_{args.prompt_type}_result.jsonl', 'a', encoding='utf-8') as fout:
        for index,i in test_file.iterrows():
            if index > 248:
                break
            claim = i['claim']
            explanation = i['explanation']
            choices = ['SUPPORT', 'REFUTE','NOT ENOUGH INFO']
            label = i['label']
            if label == "NOT ENOUGH INFO":
                try:
                    golden_entity = eval(i['result entity'])
                except:
                    golden_entity = eval(i['golden entity'])
            else:
                golden_entity = eval(i['golden entity'])
            golden_documents = ''
            for entity in golden_entity:
                entity = entity.replace('_',' ')
                cursor.execute('SELECT * FROM documents WHERE id = ?', (entity,))
                doc_summary = cursor.fetchone()
                if doc_summary:
                    doc_summary = doc_summary[1]
                    golden_documents += doc_summary
                else:
                    c = 1
                    print('no doc summary',c)
                    break
            if c == 1:
                c = 0
                continue
            if args.prompt_type == 'w_exp':
                prompt = f"Claim: {claim}\nEvidence: {golden_documents}\n Evaluate the claim based on the provided evidence and choose one of the following labels: 'SUPPORT', 'REFUTE', or 'NOT ENOUGH INFO'. Provide a brief explanation for your choice."
            elif args.prompt_type == 'claim_only':
                prompt = f"Check the claim: {claim}\nChoices: {choices}\nAnswer: "
            elif args.prompt_type == 'wo_exp':
                prompt = f"Claim: {claim}\nEvidence: {golden_documents}\n Evaluate the claim based on the provided evidence and choose one of the following labels: 'SUPPORT', 'REFUTE', or 'NOT ENOUGH INFO'."
            elif args.prompt_type == 'w_exp_doc1':
                prompt =  exp1_prompt + f"Evidence: {golden_documents}\n Check the claim: {claim}\nChoices: {choices}\nAnswer: "
            elif args.prompt_type == 'w_exp_doc3':    
                prompt =  exp3_prompt + f"Evidence: {golden_documents}\n Check the claim: {claim}\nChoices: {choices}\nAnswer: "
            elif args.prompt_type == 'json':
                prompt = f"Evidence: {golden_documents}\n Check the claim: {claim}\nChoices: {choices}\nAnswer: "
            if len(prompt.split()) >= 1200:
                continue
            succ += 1
            # cot_prompt =  base_prompt+prompt
            # print(cot_prompt)
            num_retry = 0
            print('prompt:',prompt)
            while True:
                try:
                    result = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo", 
                        messages=[ 
                        {"role": "user", "content": prompt} 
                        ] 
                    )
                    break
                except Exception as e:
                    print(e)
                    if num_retry >= 5:  # Retried too many times
                        print('Retried too many times, skip this instance.')
                        break
                    time.sleep(5)
                    num_retry += 1
                    continue
            predict = result
            print('predict:',predict)
            print('=================')
            fout.write(json.dumps({'prompt':prompt,'label':label,'predict':predict,'explanation':explanation},ensure_ascii=False))
            fout.write('\n')
            time.sleep(20)

def caculate_acc_w_o_nei(url) -> None:
    res_bad = []
    bad = []
    count = 0
    scc = 0
    with open(url, 'r', encoding='utf-8') as fr:
        for index,line in enumerate(fr):
            data = json.loads(line)
            label = data['label']
            if label == 'NOT ENOUGH INFO':
                continue
            data['predict'] = data['predict']['choices'][0]['message']['content']
            predict = data['predict'].split(".", 1)[0]
            exp_gen = data['predict'].split(".", 1)[-1]
            explanatory = ''
            if label!= predict:
                bad.append({'label':label,'predict':predict,'explanatory':explanatory,'generated explanation':exp_gen,})
                res_bad.append(data)
            else:
                count += 1
            scc += 1
    bad_csv = pd.DataFrame.from_dict(bad)
    bad_csv.sample().to_dict()
    cm = confusion_matrix(bad_csv['label'], bad_csv['predict'], labels=['SUPPORT', 'REFUTE', 'NOT ENOUGH INFO'])
    print(cm)
    print(count/scc)
    print(scc)

def caculate_acc(url) -> None:
    res_bad = []
    bad = []
    count = 0
    scc = 0
    all_predicts = []
    with open(url, 'r', encoding='utf-8') as fr:
        for index,line in enumerate(fr):
            data = json.loads(line)
            label = data['label']
            data['predict'] = data['predict']['choices'][0]['message']['content']
            predict = data['predict'].split(".", 1)[0]
            exp_gen = data['predict'].split(".", 1)[-1]
            explanatory = ''
            if label!= predict:
                bad.append({'label':label,'predict':predict,'explanatory':explanatory,'generated explanation':exp_gen,})
                res_bad.append(data)
            else:
                count += 1
            scc += 1
            all_predicts.append({'label':label,'predict':predict,'explanatory':explanatory})
    bad_csv = pd.DataFrame.from_dict(bad)
    bad_csv.sample().to_dict()
    all_predicts_csv = pd.DataFrame.from_dict(all_predicts)
    cm = confusion_matrix(bad_csv['label'], bad_csv['predict'], labels=['SUPPORT', 'REFUTE', 'NOT ENOUGH INFO'])
    print(cm)
    print(confusion_matrix(all_predicts_csv['label'], all_predicts_csv['predict'], labels=['SUPPORT', 'REFUTE', 'NOT ENOUGH INFO']))
    print(count/scc)
    print(scc)



def caculate_acc_w_o_nei_noexp(url) -> None:
    res_bad = []
    bad = []
    count = 0
    scc = 0
    all_predicts = []
    with open(url, 'r', encoding='utf-8') as fr:
        for index,line in enumerate(fr):
            data = json.loads(line)
            label = data['label']
            if label == 'NOT ENOUGH INFO':
                continue
            predict = data['predict']
            explanatory = ''
            if label!= predict:
                bad.append({'label':label,'predict':predict,'explanatory':explanatory})
                res_bad.append(data)
            else:
                count += 1
            scc += 1
            all_predicts.append({'label':label,'predict':predict,'explanatory':explanatory})
    bad_csv = pd.DataFrame.from_dict(bad)
    bad_csv.sample().to_dict()
    cm = confusion_matrix(bad_csv['label'], bad_csv['predict'], labels=['SUPPORT', 'REFUTE', 'NOT ENOUGH INFO'])

    print(cm)
    print(count/scc)
    print(scc)

def caculate_acc_noexp(url) -> None:
    res_bad = []
    bad = []
    count = 0
    scc = 0
    with open(url, 'r', encoding='utf-8') as fr:
        for index,line in enumerate(fr):
            data = json.loads(line)
            label = data['label']
            predict = data['predict']
            explanatory = ''
            if label!= predict:
                bad.append({'label':label,'predict':predict,'explanatory':explanatory})
                res_bad.append(data)
            else:
                count += 1
            scc += 1
    bad_csv = pd.DataFrame.from_dict(bad)
    bad_csv.sample().to_dict()
    cm = confusion_matrix(bad_csv['label'], bad_csv['predict'], labels=['SUPPORT', 'REFUTE', 'NOT ENOUGH INFO'])
    print(cm)
    print(count/scc)
    print(scc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--proxy', action='store_true', default=False,
                    help='Whether to use proxy')
    parser.add_argument('prompt_type', type=str, default=None,
                        help='Prompt type')
    args = parser.parse_args()
    if args.proxy:
        os.environ['https_proxy'] = 'http://127.0.0.1:7895'
        os.environ['http_proxy'] = 'http://127.0.0.1:7895'
        os.environ['all_proxy'] = 'socks5://127.0.0.1:7895'

    main()

    url = f'results/mini_{args.prompt_type}_result.jsonl'
    caculate_acc_w_o_nei(url)
    caculate_acc(url)
