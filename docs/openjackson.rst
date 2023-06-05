Model: Open Jackson Network
===============================================

Description:
------------
This model represents an Open Jackson Network with Poisson arrival time, exponential service time, and probabilistic routing.

Sources of Randomness:
----------------------
There are 3 sources of randomness in this model:
1. Exponential inter-arrival time of customers at each station.
2. Exponential service time of customers at each station.
3. Routing of customers at each station after service.

Model Factors:
--------------
* number_queues: The number of queues in the network.
    * Default: 3
* arrival_alphas: The rate parameter of the exponential distribution for the inter-arrival time of customers at each station.
    * Default: 
* routing_matrix: The routing probabilities for a customer at station i to go to service j after service. 

The departure probability from station i is :math: `1 - \sum_{j=1}^{n} (P_{ij})`

where n is the number of stations, and P is the routing matrix.
    * Default: [[0.0, 0.5, 0.0], [0.0, 0.0, 0.5], [0.5, 0.0, 0.0]]
* departure_probabilities: The probabilities for a customer to leave the network after service at each station.
// service_rates_capacity: ask if necessary

Responses:
----------
* average_queue_length: The time-average queue length at each station.
