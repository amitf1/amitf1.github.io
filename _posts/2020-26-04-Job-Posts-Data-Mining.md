---
title: "Job Posts Data Mining"
date: 2020-04-26
tags: [Data Mining, Web Scraping, Databases]
header:
  image: "/images/data-mining/jobs.jpeg"
excerpt: "Data Mining, Web Scraping, Databases"
mathjax: "true"
---


## Web Scraping Glassdoor's job posts using Selenium and creation of a DB with the relevant data.

We created a web scraper, that goes through defined job searches, and extracts relevant job posts data.
This data is then enriched and pushed into a MYSQL DB we created.

## The Data
The following images shows example of the source that was scarped.

Jobs list:
<img src="{{ site.url }}{{ site.baseurl }}/images/data-mining/GLASSDOOR1.png" alt="jobs">
Job post:
<img src="{{ site.url }}{{ site.baseurl }}/images/data-mining/GLASSDOOR2.png" alt="post">

### Data Enrichment​
The following API's were used:
- Rapid API – Find the country​ where it is missing
- Geopy​ - Find the Longitude and Latitude
- Rest Countries ​- Add additional country info

Another useful information we added to our data was skills needed for each job.
Skills mining using "Bag of Skills" - We extracted words and pairs of words (1&2-grams) from the job description and matched them to skills from our skills list.
If a match was found we add a corresponding row to our skill-job table, enabling us to analyze the skills needed withing different jobs.

### Database Design
Below is the ERD for our database.
<img src="{{ site.url }}{{ site.baseurl }}/images/data-mining/glassdoor5.png" alt="erd">

### Example for possible analysis
We can do many kinds of analysis on the DB we created, below is an example.

#### Top 20 Skills Needed for a Data Scientist within Our Data.
The bar show the count of the job posts (y-axis) related to each skill(X-axis).
<img src="{{ site.url }}{{ site.baseurl }}/images/data-mining/glassdoor6.png" alt="skills">

[Repository with the Full Code](https://github.com/amitf1/Data_Mining_Glassdoor)
