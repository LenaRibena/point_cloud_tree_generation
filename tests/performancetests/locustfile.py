from locust import HttpUser, between, task

# locust -f tests/performancetests/locustfile.py


class MyUser(HttpUser):  # type: ignore
    wait_time = between(1, 2)

    @task  # type: ignore
    def get_flow(self) -> None:
        self.client.get("/generate/flow")

    @task  # type: ignore
    def get_gauss(self) -> None:
        self.client.get("/generate/gauss")
