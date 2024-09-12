from py4j.java_gateway import JavaGateway

gateway = JavaGateway()
app = gateway.entry_point
result = app.reset()
print(result)
