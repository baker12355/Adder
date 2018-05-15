# Adder to Substractor

> My main idea is easy. 
>>   1.Generate the input sequeece :'999-888' , '  500-60' .. which len(x)=7.
>>   And the output should be :' 111' ,' 440' .. which len(x)=4.
    
>>   2.One hot encoding each them and follow the order --> space sub 0 1 2 3 4 5 6 7 8 9
>>   so '9' =  ['F F F F F F F F F F F T'] , ' ' = ['T F F F F F F F F F F F']
>>   therefore the input become 7x12 matrix and output 4x12
  
>>  3.Use RNN training
