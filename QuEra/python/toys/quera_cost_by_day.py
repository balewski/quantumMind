#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import boto3
import csv
from pprint import pprint
from datetime import datetime, timedelta
import argparse

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
    print('num scanned  days:',len(ResultsByTime))
    sum=0
    outL=[]
    for rec in ResultsByTime:
        tEnd=rec['TimePeriod']['End']
        xx=rec['Total']['AmortizedCost']
        amount=float(xx['Amount'])
        outL.append(amount)
        sum+=amount
        if amount <1.3 : continue
        print('day:%s  %8.2f %s  sum=%8.2f'%(tEnd,amount,xx['Unit'],sum))
        #print('day:%s  %.2f %s  '%(tEnd,amount,xx['Unit']))
    print('SUM %d USD'%sum)
    return outL


#=================================
#  M A I N
#=================================

if __name__ == "__main__":
        
    today = datetime.today()  # Get today's date    
    # Calculate the default start and end dates
    default_end_date = (today + timedelta(days=1)).strftime('%Y-%m-%d')
    default_start_date = (today - timedelta(days=30)).strftime('%Y-%m-%d')

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process IBM Quantum jobs data.")
    parser.add_argument("--start_date", type=str, default=default_start_date, help="Start date in YYYY-MM-DD format.")
    parser.add_argument("--end_date", type=str, default=default_end_date, help="End date in YYYY-MM-DD format.")
    args = parser.parse_args()
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    assert  datetime.strptime(args.start_date, "%Y-%m-%d") < datetime.strptime(args.end_date, "%Y-%m-%d")
    
    costL=query_cost(args.start_date, args.end_date,verb=1)

    '''  convert daily cost in USD to shots, assume
    - daily use cost below $1.3 is due to some other services
    - about 4 jobs are run per day --> subtract $1.3 
    - convert remaining USD to shots at the rate $1=100 shots , based on
    https://aws.amazon.com/braket/pricing/
    '''

    totShots=0
    for cost in costL:
        if cost <1.3: continue
        totShots+= 100*(cost-1.3)

    print('\ntotal shots %.1e over period: %s to  %s '%(totShots,args.start_date ,args.end_date))

    totSec=totShots/3  # assume 3 shots/sec
    print('total AWS time %.1f hours '%(totSec/3600)) 



