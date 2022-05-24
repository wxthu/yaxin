"""
  Convert scheduling problem to Knap problem, wherein weights refer to 
  gpu memory for detector process, prices related to latency or throughput
  
  We solve it by calling knap() and derive the optimal problem by decision()
  
  Note: items index from 1 to n, same as corresponding weight and price
  so when we define weight and price list, we traverse it from 1 rather than 0
"""
class Decision:
    def __init__(self, weights=None, prices=None, number=1, capacity=1) -> None:
        self.w = weights
        self.v= prices
        self.n = number
        self.cap = capacity
        
    def knap(self):
        dp = [[0 for _ in range(self.cap + 1)] for _ in range(self.n + 1)]
        for i in range(1, self.n + 1):
            for r in range(1, self.cap + 1):
                if r < self.w[i]:
                    dp[i][r] = dp[i-1][r]
                else:
                    dp[i][r] = max(dp[i-1][r], dp[i-1][r-self.w[i]]+self.v[i])
        
        return dp
    
    def decision(self, x=None):
        # x : boolean list, indicating whether item i will be packed
        dp = self.knap()
        i = self.n
        r = self.cap
        max_value = 0
        
        selected = []
        while i > 0:
            
            if dp[i][r] != dp[i-1][r]:
                selected.append(i)
                max_value += self.v[i]
                r = r - self.w[i]
           
            i = i - 1
        
        return selected
    
if __name__ == '__main__':
	wt = [0, 2, 2, 6, 5, 4]
	pc = [0, 6, 3, 5, 4, 6]
	num = 5
	cap = 10
	d = [0 for _ in range(num+1)]
	
	sched = Decision(wt, pc, num, cap)
	
	benefit = sched.decision(d)
	print(f'overall item prices : {benefit}')
	index = []
	for i in range(0, num+1):
		if d[i] == 1:
			index.append(i)
	print(f'selected items are : {index}')
	
 
