from polygon import RESTClient

client = RESTClient("DM1scQhwN988Q_oxTQAtEkvHkHnezUr6")

aggs = []
for a in client.list_aggs(
    "AAPL",
    1,
    "minute",
    "2024-09-09",
    "2024-09-11",
    adjusted="true",
    sort="asc",
    limit=120,
):
    aggs.append(a)

print(aggs)
