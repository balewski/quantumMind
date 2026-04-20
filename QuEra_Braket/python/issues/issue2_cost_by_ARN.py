#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import boto3
import csv
from pprint import pprint

def generate_cost_csv( start_date, end_date):
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
    for rec in ResultsByTime:
        tEnd=rec['TimePeriod']['End']
        xx=rec['Total']['AmortizedCost']
        amount=float(xx['Amount'])
        sum+=amount

        #print('day:%s  %.2f %s  sum=%.2f'%(tEnd,amount,xx['Unit'],sum))
        print('day:%s  %.2f %s  '%(tEnd,amount,xx['Unit']))
# Example usage
start_date = '2023-06-20'
end_date = '2023-10-10'
generate_cost_csv(start_date, end_date)


