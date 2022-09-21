# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 20:47:51 2022

@author: 呂星樺
"""

questions = {'who created python?: ':'A',
             'what year was Python created?: ':'B',
             'python is tributed to which comedy group: ':'C',
             'is the earth round?: ' :'A'}

options = [['A. guido van rossum', 'B. elon musk', 'C. bill gates', 'D. Mark zuckerburg'],
           ['A. 1989','B. 1991', 'C. 2000', 'D. 2016'],
           ['A. Lonely island', 'B. smosh', 'C. Monty python', 'D. snl'],
           ['A.True', 'B. False', 'C. sometimes', 'D. what is earth?']]


while play_again():
    new_game()
    
print('bye')


def new_game():
    guesses = []
    correct_guesses = 0
    question_num = 1
    
    for key in questions:
        print('---------')
        print(key)
        
        for i in options[question_num-1]:
            print(i)
            
        guess = input('Enter (A, B, C, ,D):')
        guess = guess.upper()
        guesses.append(guess)
        
        correct_guesses += check_ans(questions.get(key), guess)
        question_num +=1
    
    display_score(correct_guesses, guesses)
        
        
def check_ans(answer, guess):
    if answer == guess:
        print('correct!')
        return 1
    else :
        print('wrong')
        return 0
        
    
        
        
def display_score(correct_guesses, guesses):
    print('--------------')
    print('RESULTS')
    print('--------------')
    print('Answers: ', end ='')
    for i in questions:
        print(questions.get(i), end='')
    print()
    
    print('guesses: ', end ='')
    for i in guesses:
        print(questions.get(i), end=' ')
    print()
    
    score = int((correct_guesses/len(questions))*100)
        
    print('You score is' + str(score)+ '%')
        
        
def play_again():
    response = input('do you want to play again?: (yes or no)') 
    response = response.upper()
    if response == 'YES':
        return True
    else:
        return False


         
        