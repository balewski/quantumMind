#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import boto3
import csv
from pprint import pprint

def query_cost( start_date, end_date,verb=1):
    # Set up AWS credentials
    session = boto3.Session()
    ce_client = session.client('ce')  # AWS Cost Explorer client

    # Retrieve cost information
    response = ce_client.get_cost_and_usage(
        TimePeriod={
            'Start': start_date,
            'End': end_date
        },
        Granularity='DAILY',
        Metrics=['AmortizedCost'],
        Filter={
            'And': [
                {
                    'Dimensions': {
                        'Key': 'RECORD_TYPE',
                        'Values': ['Usage']
                    }
                }, 
                {
                    'Dimensions': {
                        'Key': 'RECORD_TYPE',
                        'Values': ['Usage']
                    }
                }, 

                
            ]
        }
    )
            

    #pprint(response)
    ResultsByTime=response['ResultsByTime']
    print('num days',len(ResultsByTime))
    sum=0
    outL=[]
    for rec in ResultsByTime:
        tEnd=rec['TimePeriod']['End']
        xx=rec['Total']['AmortizedCost']
        amount=float(xx['Amount'])
        outL.append(amount)
        sum+=amount
        if amount <1.3 : continue
        print('day:%s  %.2f %s  sum=%.2f'%(tEnd,amount,xx['Unit'],sum))
        #print('day:%s  %.2f %s  '%(tEnd,amount,xx['Unit']))
    print('SUM %d USD'%sum)
    return outL    

# Example usage
start_date = '2023-04-01'
end_date = '2024-04-10'
costL=query_cost(start_date, end_date,verb=1)

'''  convert daily cost in $$$ to shots, assume
- daily use cost below $1.3 is due to some other service
- about 3 jobs are run per day --> subtract $1.3  (~4 jobs)
- convert remaining $$$ to shots at the rate $1=100 shots , based on
https://aws.amazon.com/braket/pricing/
'''

totShots=0
for cost in costL:
    if cost <1.3: continue
    totShots+= 100*(cost-1.3)

print('\ntotal shots %.1e over perido %s to  %s '%(totShots,start_date ,end_date))

totSec=totShots/3  # assume 3 shots/sec
print('total AWS time %.1f hours '%(totSec/3600)) 



