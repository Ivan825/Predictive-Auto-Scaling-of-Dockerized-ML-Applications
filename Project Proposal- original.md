# 1. Project Proposal- original

Author: Ivan Bhargava
Status: Published
Category: Proposal

# **Project Proposal**

## **Project Title:**

**Predictive Auto-Scaling of Dockerized Machine Learning Applications for Smart Manufacturing**

## **Background**

**This work is an extension of the work described in "Design and Implementation of Cloud Docker Application Architecture Based on Machine Learning in Container Management for Smart Manufacturing" [(Kim et al., 2022)](https://www.mdpi.com/1709576), which investigated the deployment of machine learning models in Docker containers on cloud infrastructure for the detection of manufacturing defects. The initial paper had shown enhanced resource utilization over non-containerized arrangements but lacked the inclusion of automated or predictive scaling.**

## **Objective**

**The main goal of this project is to develop, deploy, and test a predictive auto-scaling system for Docker containerized ML applications in smart manufacturing. We will enhance resource utilization and system performance by predicting workload and scaling the number of containers dynamically.**

### **Scope of Work**

**This project only deals with improving the container management layer of the original paper and does not involve edge deployment. The major deliverables are:**

**1.Recreate and verify the original Docker-based ML deployment for defect detection on a cloud platform (e.g., AWS EC2).**

**2.Design a predictive auto-scaling mechanism that predicts resource utilization (e.g., CPU, memory, and request rate) and scales the number of active containers accordingly.**

1. **Train system metrics and request log-based time-series forecasting models (Prophet and LSTM) to support proactive container scaling.**
2. **Compare the predictive scaling system with the baseline static/manual scaling method employed in the original paper.**
3. **Measure system performance based on CPU/memory usage, response latency, and scalability across different simulated loads.**

### **Prediction Strategy**

**The project will forecast future system demand—e.g., inference request volume and resource utilization—using time-series forecasting models:**

- **Prophet: Ideal for data with trend and seasonality components; simple to interpret and works well with sparse training data.**
- **LSTM: A deep learning algorithm that can learn intricate patterns in sequential data; ideal for predicting resource utilization trends.**

**The models will be trained on logs gathered through tools such as Docker stats or DataDog, which give timestamped logs of container resource usage and rate of service calls.**

### **Data and Experimentation**

**The predictive models will be trained on one or more of the following datasets:**

- **Synthetic Logs: Artificial data mimicking usage peaks, daily cycles, and idle times.**
- **Live Docker Metrics: Gathered from actual containers in a test environment.**
- **Public Traces: Like Google Cluster Trace or Alibaba Cluster Trace for additional analysis.**

**Trained models will make predictions of near-term resource demands (e.g., the subsequent 15–60 minutes), and the system will apply such predictions to make decisions about adding or removing containers via Docker commands or orchestration APIs.**

### **Automation Objectives**

**The system will automate the below tasks:**

- **Scaling Up: Start running new containers ahead of actual demand based on demand forecasts.**
- **Scaling Down: Stop/remove containers when demand is predicted to reduce.**

**This facilitates more efficient resource use, quicker response under heavy load, and saving on cost when idle.**

## **Evaluation Plan**

**The performance will be evaluated on two configurations:**

1. **Baseline: Manual/static scaling (as done in the baseline paper).**
2. **Improved: Auto-scaling system based on predictive methods with Prophet and LSTM.**

**Critical parameters to evaluate:**

- **CPU and memory usage**
- **Number of containers deployed over time**
- **Handling latency of requests**
- **Accuracy of predicted vs actual resource requirement**

## **Deployment of Predictive Scaling Component**

**To keep the architecture modular and consistent, the predictive scaling model (either based on Prophet or LSTM) will also be containerized. The model will be bundled into a standalone Docker container so that it can execute on its own on the same host or on a companion cloud service.**

**The container will:**

- **Execute on a schedule (e.g., every 5–10 minutes) to predict anticipated system load.**
- **Access logs of resource usage or live metrics from monitoring tools or shared volumes.**
- **Output scaling actions (e.g., scale to 4 containers, scale to 1).**
- **Run Docker commands or API calls to trigger scaling actions.**

**This container will have proper permissions to communicate with the Docker Engine or orchestration APIs safely. Running it in a containerized form ensures improved portability, isolation, and maintainability, which aligns with the overall project architecture.**

### **What If the Forecast Is Incorrect?**

**Two principal types of prediction mistakes:**

1. **Overprediction (False Alarm)**
- **The model believes high demand is ahead — but it isn't.**
- **System generates too many containers.**
- **Outcome: Wasted resources / more expense, but system functions.**
1. **Underprediction (Missed Load)**
- **The model doesn't predict the surge.**
- **Demand outstrips containers available.**
- **Outcome: System overload, slower response, potential dropped requests.**

**How the System Responds To This**

1. **Real-time Correction (Reactive Safety Net)**

**Although the system is predictive, would need real-time monitoring to:**

- **Detect when a prediction was incorrect**
- **Reactively scale containers**

**For instance:**

- **CPU reaches 90% suddenly → system fires up additional containers, even though the forecast did not indicate it.**
- **Service remains alive, and it learns from the error.**
1. **Continuous Learning / Model Updating**

**The predictive model can learn from errors, but not immediately.**

- **We can retrain Prophet/LSTM every week or day on new logs that contain the error cases.**
- **So the next time, it should perform better under the same conditions.**

**But:**

- **Prophet isn't online-learning based — you have to retrain on occasion.**
- **LSTM can be used for incremental learning but must be carefully tuned.**

**Can automate this model retraining process as part of a routine pipeline (e.g., every day at midnight).**

| **Approach** | **Complexity** | **Practicality** | **Research Value** | **Recommended?** |
| --- | --- | --- | --- | --- |
| 1. **Simple Ceiling Logic**
ceil(pred / capacity) | Easy | Used in production | Good baseline | Include as baseline |
| 2. **Buffered Logic**
ceil((pred × buffer) / capacity) | Medium | Safer against spikes | Realistic & improvable | Recommend as main logic |
| 3. **Confidence-Aware Scaling**
ceil(yhat_upper / capacity) | Medium | Conservative | Research value | Strong alternate logic |
| 4. **Policy-Aware Threshold Scaling**
(with cooldown, scale-down thresholds) | Medium+ | Industry-level polish | Publishable robustness | Best overall for paper |
| 5. **Reinforcement Learning** | Hard | Hard to tune, high setup | Novel, but complex | Only if paper is purely about RL scaling |

# When Is Each Scaling Policy Effective?

| Policy Type | Best Used When... | Weakness / Trade-off |
| --- | --- | --- |
| **1. Simple Ceiling**`ceil(pred / capacity)` |  Workload is smooth and predictable Forecasts are reasonably accurate |  Over-scales when forecast spikes or noisy |
| **2. Buffered Logic**`ceil((pred × buffer) / capacity)` |  Forecasts are okay, but occasional spikes exist Traffic shows burstiness |  May still waste resources during dips |
| **3. Confidence-Aware**`ceil(yhat_upper / capacity)` |  Forecast model gives uncertainty bound Better to **over-provision than under-provision** (e.g., finance, trading) |  Often results in over-scaling |
| **4. Policy-Aware Threshold Scaling**(cooldown, min/max, hysteresis) |  You want **stability**, not rapid scaling Latency cost is tolerable Cost control is a priority |  Might lag behind fast-changing workloads |
| **5. Reinforcement Learning** (RL directly controlling scaling) |  Complex trade-offs between latency and cost Dynamic environments Long-running learning is possible |  Slow to converge; harder to debug |

---