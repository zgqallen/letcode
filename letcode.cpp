// letcode.cpp : Defines the entry point for the console application.
//

#include <fstream>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <type_traits>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stddef.h>
#include <direct.h>
#include <iomanip>
#include <cmath>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#include <algorithm>
#include <sys/timeb.h>
#include <sys/types.h>        /*  socket types              */
#pragma comment (lib, "Ws2_32.lib")

using namespace std;

/* 1. Two Sum */
vector<int> twoSum(vector<int>& nums, int target) {
{
    unordered_map<int, int> imap;
    
    for (int i = 0; i < nums.size(); ++i) {
        auto it = imap.find(target - nums[i]);
        
        if (it != imap.end()) 
		{
			vector<int> res;
			res.push_back(i);
			res.push_back(it->second);
			return res;
		}
            
        imap[nums[i]] = i;
    }
}

/* 15. 3Sum */
vector<vector<int>> threeSum(vector<int>& nums) {
	vector<vector<int>> result;
	int len = int(nums.size() - 1);

	if(nums.size() < 3)
        return result;

	sort(nums.begin(), nums.end());
	for(int li = 0; li < len; li++)
	{
		int mi, ri;
		mi = li + 1;
		ri = len;

		if(nums[li] > 0 || nums[ri] < 0) break;
		if(li > 0 && nums[li] == nums[li-1]) continue;

		while(mi < ri)
		{
			int sum = nums[li] + nums[mi] + nums[ri];

			if(sum == 0)
			{
				/* we got a sulotion */
				vector<int> res;
				int mi_v = nums[mi];
				int ri_v = nums[ri];

				res.push_back(nums[li]);
				res.push_back(nums[mi]);
				res.push_back(nums[ri]);
				result.push_back(res);
				mi++;
				ri--;
				while(mi_v == nums[mi] && mi < ri) mi++;
				while(ri_v == nums[ri] && ri > mi) ri--;
			}
			else
			{
				sum > 0 ? ri-- : mi++;
			}
		}
	}

	return result;
}

/* 16. 3Sum Closet*/
int threeSumClosest(vector<int>& nums, int target)
{
	int res = 0, dis = INT_MAX;
	int len = int(nums.size() - 1);

	sort(nums.begin(), nums.end());
	for(int li = 0; li < len; li++)
	{
		int mi = li + 1, ri = len;
		while(mi < ri)
		{
			int sum = nums[li] + nums[mi] + nums[ri];
			int t_dis = abs(sum - target);
			if(t_dis < dis)
			{
				dis = t_dis;
				res = sum;
			}

			if(sum == target)
			{
				return sum;
			}
			else 
			{
				sum > target ? ri-- : mi++;
			}
		}
	}

	return res;
}


int main(int argc, char* argv[])
{

	int arry[] = {0,0,0,0};
	vector<int> in(arry, arry+4);
	threeSum(in);

	system("pause");

	return 0;
}

