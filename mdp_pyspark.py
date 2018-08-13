from pyspark import SparkContext
from pyspark.conf import SparkConf
import numpy as np
from pyspark.ml.linalg import Vectors
from pyspark.mllib.random import RandomRDDs
from pyspark.sql.types import *
from pyspark.sql import DataFrame
from pyspark import SparkConf, SparkContext
import pyspark.sql.functions as func
from pyspark.sql.functions import *
from pyspark.sql.types import StringType, IntegerType, StructType, StructField, DoubleType, FloatType, DateType, TimestampType
from pyspark.sql.functions import date_format, col, desc, udf, from_unixtime, unix_timestamp, date_sub, date_add, last_day
from pyspark.sql.functions import round, sum, lit, add_months, coalesce, max
from pyspark.sql.functions import rand, randn
from collections import OrderedDict
from pyspark.sql import Row
import random
from datetime import datetime
import time

sc.addPyFile("maprfs:///user/mapr/tmp/agent.py")
from agent import Agent

num_agents = 3
agentRDD = sqlContext.createDataFrame(zip(range(1, num_agents + 1)), ["equipment_id"]).rdd
agentRDD.take(10)

type(agentRDD)

def create_agent(row):
    agent = Agent(row)
    agent.current_state = 'good'
    agent.active = True
    agent.total_benefit = 1000
    agent.total_cost = 0
    agent.maintenance_performed = 'N'
    agent.number_maintenance_performed = 0
    agent.replacement = 'N'
    agent.number_replacements = 0
    agent.policy = 0
    return agent

def update_status(row):
    agent = row
    p = random.uniform(0,1)
    #agent.transition_prob = p
    if agent.current_state == 'good' and agent.replacement == 'N':     
        if p < 0.9 and p > .6:
            agent.current_state = 'acceptable'
            agent.total_benefit += 500
        elif p > .9:
            agent.current_state = 'bad'
            agent.total_cost += 500
        else:
            agent.current_state = 'good'
            agent.total_benefit += 1000
    elif agent.current_state == 'good' and agent.replacement == 'Y':
        agent.current_state = 'good'
        agent.total_benefit += 1000
        agent.replacement = 'N'
    elif agent.current_state == 'good' and  agent.maintenance_performed == 'Y':
        agent.current_state = 'good'
        agent.total_benefit += 1000
        agent.maintenance_performed = 'N'
    elif agent.current_state == 'acceptable':
        if p > 0.6:
            agent.current_state = 'bad'
            agent.total_cost += 500
        else:
            agent.current_state = 'acceptable'
            agent.total_benefit += 500
    else:
        agent.current_state = 'bad'
        agent.total_cost += 500
    return agent


def policy_one(row):
    agent = row
    #agent.policy = 1
    if agent.current_state == 'bad':    
        agent.replacement = 'Y'
        agent.number_replacements += 1
        agent.total_cost += 2000                   #replacement cost, immediately brings equipment back to good condition
        agent.current_state = 'good'
    return agent

def policy_two(row):
    agent = row
    #agent.policy = 2
    if agent.current_state == 'acceptable':
        agent.mainteance_performed = 'Y'
        agent.number_maintenance_performed += 1
        agent.total_cost += 500                    #cost of fixing acceptable equipment
        agent.current_state = 'good'
    if agent.current_state == 'bad': 
        agent.replacement = 'Y'
        agent.number_replacements += 1
        agent.total_cost += 2000                   #replacement cost, immediately brings equipment back to good condition
        agent.current_state = 'good'
    return agent


def policy_three(row):
    agent = row
    #agent.policy = 3
    if agent.current_state == 'acceptable' or agent.current_state == 'bad':    
        agent.replacement = 'Y'
        agent.number_replacements += 1
        agent.total_cost += 2000                   #replacement cost, immediately brings equipment back to good condition
        agent.current_state = 'good'
    return agent


def mdp_simulation(row,time,policy):
    results = []
    agent = create_agent(row)
    for week in range(time):
        if week < 1:
            results.append((week,
                            agent.id,
                            agent.current_state, 
                            agent.total_benefit, 
                            agent.total_cost,
                            agent.maintenance_performed,
                            agent.number_maintenance_performed,
                            agent.replacement,
                            agent.number_replacements))
        else:
            update_status(agent)
            results.append((week,
                            agent.id,
                            agent.current_state, 
                            agent.total_benefit, 
                            agent.total_cost,
                            agent.maintenance_performed,
                            agent.number_maintenance_performed,
                            agent.replacement,
                            agent.number_replacements))
            policy(agent)
    return results


cols = ("Week", "id", "current_state", "total_benefit", "total_cost", 
        "maintenance_performed", "number_maintenance_performed",
        "replacement","number_replacements")
t = 100
policy_One_results = agentRDD.flatMap(lambda a: mdp_simulation(a,t,policy_one)).toDF().orderBy('_1','_2').toDF(*cols)
policy_One_results.show(100,False)



policy_One_summary = policy_One_results.select([
                                                func.sum(policy_One_results.total_benefit).alias("Total_Gains"),
                                                func.sum(policy_One_results.total_cost).alias("Total_Cost"),
                                                max(policy_One_results.number_replacements).alias("Number_of_Replacements")])
policy_One_summary.show()

policy_Two_results = agentRDD.flatMap(lambda a: mdp_simulation(a,t,policy_two)).toDF().orderBy('_1','_2').toDF(*cols)
policy_Three_results = agentRDD.flatMap(lambda a: mdp_simulation(a,t,policy_three)).toDF().orderBy('_1','_2').toDF(*cols)

policy_Two_summary = policy_Two_results.select([
                                                func.sum(policy_Two_results.total_benefit).alias("Total_Gains"),
                                                func.sum(policy_Two_results.total_cost).alias("Total_Cost"),
                                                max(policy_Two_results.number_replacements).alias("Max_Number_of_Replacements")])
policy_Two_summary.show()


policy_Three_summary = policy_Three_results.select([
                                                func.sum(policy_Three_results.total_benefit).alias("Total_Gains"),
                                                func.sum(policy_Three_results.total_cost).alias("Total_Cost"),
                                                max(policy_Three_results.number_replacements).alias("Max_Number_of_Replacements")])
policy_Three_summary.show()
