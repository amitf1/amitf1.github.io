---
title: "Job Posts Data Mining"
date: 2020-04-26
tags: [Data Mining, Web Scraping, Databases]
header:
  image: "/images/data-mining/data.png"
mathjax: "true"
---


## Web Scraping Glassdoor's job posts using selenium, and creation of a DB with the relevant data.

We created a web scraper, that goes through defined job searches, and extracts relevant job posts data.
This data is then enriched and pushed into a MYSQL DB we created.

## The Data
The following images shows example of the source that was scarped.

Jobs list:
<img src="{{ site.url }}{{ site.baseurl }}/images/data-mining/GLASSDOOR1.png" alt="jobs">
Job post:
<img src="{{ site.url }}{{ site.baseurl }}/images/data-mining/GLASSDOOR2.png" alt="post">

### Data Enrichment​
- Rapid API – Find the country​ where it is missing
- Longitude  and Latitude using Geopy​
- Rest Countries ​- additional country info
- Skills mining with "bag of skills" - We extracted "2-grams" from the job description and matched them to skills from our skills list. (example shown below)

<img src="{{ site.url }}{{ site.baseurl }}/images/data-mining/glassdoor3.png" alt="skills">
You can see how this one post, has many skills in it's description, and we add each skill to our table.

### Database Design
Below is the ERD for our database.
<img src="{{ site.url }}{{ site.baseurl }}/images/data-mining/glassdoor5.png" alt="erd">

## Example for possible analysis
We can do many analysis on the DB we created, below is an example, how we found the 20 top needed skills for a Data Scientist within our data.
<img src="{{ site.url }}{{ site.baseurl }}/images/data-mining/glassdoor6.png" alt="skills">

[Repository with the Full Code](https://github.com/amitf1/Data_Mining_Glassdoor)
