// letcode.cpp : Defines the entry point for the console application.
//

#include <fstream>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <queue>
#include <set>
#include <unordered_set>
#include <map>
#include <stack>
#include <cstdint>
#include <type_traits>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stddef.h>
#include <direct.h>
#include <iomanip>
#include <cmath>
#include <string>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#include <algorithm>

using namespace std;

/* 1. Two Sum */
vector<int> twoSum(vector<int>& nums, int target) {
	vector<int> res;
	unordered_map<int, int> imap;
    
	for (int i = 0; i < nums.size(); ++i) {
        auto it = imap.find(target - nums[i]);
        
		if (it != imap.end()) 
		{
			res.push_back(i);
			res.push_back(it->second);
			return res;
		}
            
		imap[nums[i]] = i;
	}

	return res;
}

/* 2. Add Two Numbers*/
struct ListNode {
	int val;
	ListNode *next;
	ListNode(int x) : val(x), next(NULL) {}
};
ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        ListNode *p1, *p2, *cur = NULL, *pre = NULL, *ret = NULL;
        int val = 0, inc = 0;
        unsigned int sum = 0;
        
        p1 = l1;
        p2 = l2;
        
        ret = pre = new ListNode(0);
        while(true)
        {          
            if(p1 || p2 || inc)
            {
                val = ((p1 != NULL)?p1->val:0) + ((p2 != NULL)?p2->val:0) + inc;
            }
            else
            {
                break;
            }
            
            inc = val/10;
            val = val - inc*10;
            
            cur = new ListNode(val);

            pre->next = cur;
            pre = cur;
            
            if(p1 != NULL) p1 = p1->next;
            if(p2 != NULL) p2 = p2->next;
        }
        
        cur = ret;
        ret = cur->next;
        delete cur;
        
        return ret; 
}

/* 3. Longest Substring Without Repeating Characters */
int lengthOfLongestSubstring(string s) {
	int bit[256] = {-1}; 
	int i, last = -1, longest = 0;

	memset(bit, -1, sizeof(bit));
	for(i = 0; i < s.size(); i++)
    {
		if(bit[s[i]] != -1) last = (last > bit[s[i]])?last:bit[s[i]];

		bit[s[i]] = i;
		longest = (i - last) > longest? (i - last):longest;
	}

	return longest;
}

/* 4. Median of Two Sorted Arrays*/
double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
       int idx = 0, total, median, i = 0, j = 0, val1 = 0, val2 = 0;
        
        total = nums1.size() + nums2.size();
        median = (total + 1)/2;
        
        i = nums1.size();
        j = nums2.size();
        
        if(i == 0 || j == 0)
        {
            (i == 0)?val1 = nums2[j-1]:val1 = nums1[i-1];
        }
        
        while(true)
        {
            if(i && j)
            {
                (nums1[i-1] >= nums2[j-1])? (val2 = val1, val1 = nums1[i-1], i--): (val2 = val1, val1 = nums2[j-1], j--);
            }
            else if(i)
            {
                val2 = val1;
                val1 = nums1[i-1];
                i--;
            }
            else
            {
                val2 = val1;
                val1 = nums2[j-1];
                j--;
            }
            
            idx++;
            if((idx == (median + 1)) || (idx >= total))  break;
        }
        
        return (total % 2)?(val2*1.0):(((val1+val2)*1.0)/2);
}

/* 5. Longest Palindromic Substring */
string longestPalindrome(string s) {
		const int size = s.size();
		int **dp = new int*[size];
        int i, j, l = 0, r = 0, longest = -1;

        for(i = 0; i < size; i++)
			dp[i] = new int[size];

		for(i = 0; i < size; i++)
		{
			for(j = 0; j <size; j++)
				dp[i][j] = 0;
		}

        for(i = 0; i < size; i++)
        {
            for(j = 0; j <= i; j++)
            {
                dp[j][i] = ((s[j] == s[i]) && ((i - j) < 2 || dp[j+1][i-1]));
                if(dp[j][i] && longest < (i - j + 1))
                {
                    longest = i - j + 1;
                    l = j;
                    r = i;
                }
            }
        }
        
        return s.substr(l, r - l + 1);
}

/* 6. ZigZag Conversion */
string convert(string s, int nRows) {
    if (nRows <= 1)
        return s;

    const int len = (int)s.length();
    string *str = new string[nRows];

    int row = 0, step = 1;
    for (int i = 0; i < len; ++i)
    {
        str[row].push_back(s[i]);

        if (row == 0)
            step = 1;
        else if (row == nRows - 1)
            step = -1;

        row += step;
    }

    s.clear();
    for (int j = 0; j < nRows; ++j)
    {
        s.append(str[j]);
    }

    delete[] str;
    return s;
}

/* 7. Reverse Integer */
int reverse(int x) {
    long answer = 0;
    while (x != 0) {
       answer = answer * 10 + x % 10;
       if (answer > INT_MAX || answer < INT_MIN) return 0;
       x /= 10;
	}
    
    return (int)answer;
}

/* 8. String to Integer (atoi) */
int myAtoi(string str) {
        long result = 0;
        int indicator = 1;
        for(int i = 0; i<str.size();)
        {
            i = str.find_first_not_of(' ');
            if(str[i] == '-' || str[i] == '+')
                indicator = (str[i++] == '-')? -1 : 1;
            
            while('0'<= str[i] && str[i] <= '9') 
            {
                result = result*10 + (str[i++]-'0');
                if(result*indicator >= INT_MAX) return INT_MAX;
                if(result*indicator <= INT_MIN) return INT_MIN;                
            }
            
            return result*indicator;
        }
        
        return result;
}

/* 9. Palindrome Number */
bool isPalindrome(int x) {
        if(x < 0) return false;
        if(x < 10) return true;
        if(x % 10 == 0) return false;
        
        int rev = 0;
        
        while(rev < x)
        {
            rev = rev*10 + x%10;
            x = x/10;
        }

        return (rev == x || rev/10 == x);
}

/* 11. Container With Most Water */
int maxArea(vector<int>& height) {
        int maxarea = 0, i = 0, j = height.size() - 1;
        
        while(i < j)
        {
            int h = min(height[i], height[j]);
            maxarea = max(maxarea, h * (j - i));
            while(height[i] <= h && i < j) i++;
            while(height[j] <= h && i < j) j--;
        }
        
        return maxarea;
}

/* 14. Longest Common Prefix*/
struct TrieNode{
	int count;
	char word;
	vector<TrieNode> child;
	TrieNode():word('@'), count(0) { }
};

class Trie
{
public:
	TrieNode *root;
    unsigned int obj_c;

public:
	Trie():obj_c(0) { root = new TrieNode(); }
	~Trie() {delete root;}

    void InsertTrie(const string word)
    {
        TrieNode* pLoc = root;
        obj_c++;
        
	    for(int i = 0; i < word.size(); i++)
	    {
		    bool exist = false;

		    for(unsigned int j = 0; j < pLoc->child.size(); j++)
		    {
			    if(pLoc->child[j].word == word[i])
			    {
				    pLoc->child[j].count++;
				    pLoc = &(pLoc->child[j]);
				    exist = true;
				    break;
			    }
		    }

		    if(!exist)
		    {
			    TrieNode temp;
			    temp.word = word[i];
			    temp.count++;
			    pLoc->child.push_back(temp);
			    pLoc = &pLoc->child.back();
		    }
	    }
    }
    
    string LongestCommon()
    {
	    string common = "";

	    if(root->child.size() == 1)
	    {
		    TrieNode *pLoc = &root->child[0];
		    while(pLoc != NULL)
		    {
                if(pLoc->count != this->obj_c)
				    break;
			    common.append(&pLoc->word, 1);
			    if(pLoc->child.size() != 1)
				    break;
			    pLoc = &pLoc->child[0];
		    }
	    }

	    return common;
    }
};

string longestCommonPrefix(vector<string>& strs) {
        Trie T;
        
        for(unsigned int i = 0; i < strs.size(); i++)
        {
            T.InsertTrie(strs[i]);
        }
        
        return T.LongestCommon();
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

/* 378. Kth Smallest Element in a Sorted Matrix */
class KTOP_MAXHEAP
{
private:
	int _size;
	int max_size;
	priority_queue<int> pq;

public:
	KTOP_MAXHEAP(int K):_size(0) {max_size = K;}
	void show_queue();
	void push_queue(int v);
	int top() {return pq.top();}
};

void KTOP_MAXHEAP::show_queue()
{
	while(!pq.empty())
	{
		cout<<pq.top()<<" ";
		pq.pop();
	}
	cout<<endl;
}

void KTOP_MAXHEAP::push_queue(int v)
{
	if(pq.size() < max_size){
		pq.push(v);
	}else{
		if(v < pq.top())
		{
			pq.pop();
			pq.push(v);
		}
	}
}

int kthSmallest(vector<vector<int>>& matrix, int k) {
	KTOP_MAXHEAP k_max(k);
	int i,j,v = 0;
	for(i = 0 ; i < matrix.size(); i++)
	{
		for(j = 0; j < matrix[i].size(); j++)
		{
			k_max.push_queue(matrix[i][j]);
		}
	}
	v = k_max.top();
	cout << "Min " << k << "th value is: " << v << endl;
	k_max.show_queue();
	return v;
/* another simple using priority_queue
        priority_queue<int> q;
        for (int i = 0; i < matrix.size(); ++i) {
            for (int j = 0; j < matrix[i].size(); ++j) {
                q.emplace(matrix[i][j]);
                if (q.size() > k) q.pop();
            }
        }
        return q.top();
*/
}

/* 821. Shortest Distance to a Character */
void updateDist(vector<int> &dist, int le, int re, int side)
{
	if(side == 1){
		for(int i = 0; i < re; i++)
		{
			dist[i] = re - le - i;
		}
	}
	else if(side == 2){
		for(int i = le; i <= re; i++)
		{
			dist[i-1] = (re -le) - (re - i); 
		}
	}
	else{
		for(int i = le; i <= re; i++)
		{
			dist[i-1] = min(i - le, re - i);
		}
	}
}
vector<int> shortestToChar(string S, char C) {
	vector<int> e_idxs;
	vector<int> dist;
	int i;

	for(i = 0; i < S.length(); i++)
	{
		dist.push_back(0);
		if(S[i] == C) e_idxs.push_back(i+1);
	}

	if(e_idxs[0] != 1) updateDist(dist, 1, e_idxs[0], 1);
	if(e_idxs[e_idxs.size()-1] != dist.size()) updateDist(dist, e_idxs[e_idxs.size()-1], dist.size(), 2);

	for(i = 0; i < (e_idxs.size()-1); i++)
	{
		updateDist(dist, e_idxs[i], e_idxs[i+1], 0);
	}

	return dist;
}

/* 822. Card Flipping Game */
int flipgame(vector<int>& fronts, vector<int>& backs) {
	int small_good = 2001, back = 0;
	for(int i = 0; i < fronts.size(); i++)
	{
		int j;
		if(fronts[i] == backs[i]) continue;
		for(j = 0; j < fronts.size(); j++)
		{
			if(j == i) continue;
			if(fronts[j] == fronts[i]) break;
		}
		if(j == fronts.size() && (fronts[i] < small_good)) small_good = fronts[i];
	}
	return (small_good < 2001)?small_good:0;
}

/* 830. Positions of Large Groups */
    int recurse(string &S, int s_idx)
    {
        if(s_idx == (S.length()-1))
            return s_idx; 
            
        if(S[s_idx+1] == S[s_idx])
            return recurse(S, s_idx+1);
        else
            return s_idx;
    }
    
    vector<vector<int>> largeGroupPositions(string S) {
        vector<vector<int>> result;
        
		if(S.length() > 3)
		{
			for(int s_idx = 0; s_idx < S.length();)
			{
				int e_idx = recurse(S, s_idx);
				if((e_idx - s_idx) >= 2)
				{
					vector<int> V;
					V.push_back(s_idx);
					V.push_back(e_idx);
					result.push_back(V);
				}
				s_idx = e_idx + 1;
			}
		}
   
        return result;
}

/* 845. Longest Mountain in Array */
int longestMountain(vector<int>& A) {
	int ret = 0;

	if(A.size() < 3) return ret;
	for(int i = 1; i < (A.size()-1); i++)
	{
		int li, ri, loop = 1;

		if(A[i] > A[i-1] && A[i] > A[i+1])
		{
			li = i; ri = i;
		}
		else
		{
			continue;
		}

		while(loop)
		{
			loop = 0;
			if(li > 0 && A[li] > A[li-1])
			{
				li--; loop += 1;
			}
			if(ri < (A.size()-1) && A[ri] > A[ri+1])
			{
				ri++; loop += 1;
			}
		}

		if(ret < (ri - li + 1) && (ri - li + 1) >= 3) ret = ri - li + 1;
	}

	return ret;
}

/* 139. Word Break */
bool wordBreak(string s, vector<string>& wordDict) {
	int min_word = INT_MAX;
	set<string> DictSet;

	for(vector<string>::iterator word = wordDict.begin(); word != wordDict.end(); ++word)
	{
		DictSet.insert(*word);
		min_word = min(min_word, (int)(*word).size());
	}

	if(s.size() < min_word)
		return false;

	vector<bool> Break(s.size(), false);
	for(int i = min_word-1; i < s.size(); i+=min_word)
	{
		for( int j = i - 1; j >= -1; j--)
		{
			if(j == -1 || Break[j])
			{
				if(DictSet.find(s.substr(j+1, i-j)) != DictSet.end())
				{
					Break[i] = true;
					break;
				}
			}
		}
	}

	return Break[s.size()-1];
}

/* 647. Palindromic Substrings */
bool isPalindromic(string s)
{
	int li, ri;
	/* cout << s << endl; */
	if(s.size() == 1) return true;
	li = 0;
	ri = s.size() - 1;
	while(li <= ri)
	{
		if(s[li] == s[ri])
		{
			li++;ri--;
		}
		else
		{
			return false;
		}
	}
	return true;
}

int countSubstrings(string s) {
        int li, i, count = 0;
		/* the substr word count */
		for(i = 1; i <= s.size(); i++)
		{
			for(li = 0; li <= s.size() - i; li++)
			{
				if(isPalindromic(s.substr(li, i))) count++;
			}
		}

	return count;
}

/* 617. Merge Two Binary Trees */

struct TreeNode {
	int val;
	TreeNode *left;
	TreeNode *right;
	TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};

TreeNode* mergeTrees(TreeNode* t1, TreeNode* t2) {
	if(!t1 && !t2) return nullptr;
	if(!t1) return t2;
	else if(!t2) return t1;
	else
	{
		TreeNode *newnode = new TreeNode(t1->val + t2->val);
		newnode->left = mergeTrees(t1->left, t2->left);
		newnode->right = mergeTrees(t1->right, t2->right);
		return newnode;
	}
}

/* 581. Shortest Unsorted Continuous Subarray */
int findUnsortedSubarray(vector<int>& nums) {
        int start = -1, end = -1,  max = INT_MIN;
        for(int i=0; i<nums.size();i++){
            if(max>nums[i]){
                if(start == -1)start = i-1;
                while(start-1>=0 && nums[start-1]>nums[i])start--;
                end = i+1;
            }
            else max = nums[i];
        }
        return end - start;   
}

/* 146. LRU Cache */
class LRUCache {
public:
    LRUCache(int capacity) {
        max_size = capacity;
    }
    
    int get(int key) {
		int value = -1;
		if(LRUMap.find(key) != LRUMap.end())
		{
			value = LRUMap[key]->second;
			LRUList.erase(LRUMap[key]);
			LRUList.push_front(make_pair(key,value));
			LRUMap[key] = LRUList.begin();
		}

        return value;
    }
    
    void put(int key, int value) {
        if(LRUMap.find(key) != LRUMap.end())
		{
			LRUList.erase(LRUMap[key]);
		}
		else if(LRUMap.size() == max_size)
		{
			LRUMap.erase(LRUList.back().first);
			LRUList.pop_back();
		}

		LRUList.push_front(make_pair(key,value));
		LRUMap[key] = LRUList.begin();
    }
private:
	size_t max_size;
	list<pair<int, int>> LRUList;
	unordered_map<int, list<pair<int,int>>::iterator> LRUMap;
};

/* 44. Wildcard Matching */
bool isMatch(string s, string p) {
	int ps = p.size();
	int ss = s.size();
	/* DB solution to record matching result of s[i] to p[j]*/
	vector<vector<bool>> dp(ss+1, vector<bool>(ps+1, false));
	dp[0][0] = true;

	/* scan from p to match s */
	for(int j = 1; j <= ps; j++)
	{
		if(p[j-1] == '*')
		{
			int i = 0;
			/* any substr if s[0->i] could match p[0->j-1], then p[i][j->end] match */
			while(i <= ss)
			{
				if(dp[i][j-1]) break;
				i++;
			}

			while(i <= ss) dp[i++][j] = true;
		}
		else if(p[j-1] == '?')
		{
			for(int i = 1; i <= ss; i++)
			{
				if(dp[i-1][j-1]) dp[i][j] = true;
			}
		}
		else
		{
			for(int i = 1; i <= ss; i++)
			{
				if((p[j-1] == s[i-1]) && dp[i-1][j-1]) dp[i][j] = 1;
			}
		}
	}

   for (int i=0; i<=ss; i++){
        for (int j=0; j<=ps; j++)
            cout<<dp[i][j]<<" ";
        cout<<endl;
    }
	return dp[ss][ps];
}

/* 10. Regular Expression Matching */
bool isMatch2(string s, string p) {
	int ss = s.size();
	int ps = p.size();

	/* DB solution to record matching result of s[i] to p[j]*/
	vector<vector<bool>> dp(ss+1, vector<bool>(ps+1, false));
	dp[0][0] = true;

	/* s could be empty, so start from index 0. */
	for(int i = 0; i <= ss; i++)
	{
		for(int j = 1; j <= ps;j++)
		{
			if(p[j-1] == '*')
			{
				/* star can repeat zero or repeat many times in case p[j-2] == s[i-1]||p[j-2] == '.' */
				dp[i][j] = dp[i][j-2] || ((i >=1) && (p[j-2] == s[i-1]||p[j-2] == '.') && dp[i-1][j]);
			}
			else
			{
				dp[i][j] = (i >=1) && (p[j-1] == s[i-1]||p[j-1] == '.') && dp[i-1][j-1];
			}
		}
	}

	for (int i=0; i<=ss; i++){
        for (int j=0; j<=ps; j++)
            cout<<dp[i][j]<<" ";
        cout<<endl;
    }
	return dp[ss][ps];
}

/* 140. Word Break II */
vector<string> result;
void WBS(vector<vector<int>> &dp, int s_idx, string s, vector<string> &res)
{
	for(int i = 0; i < dp[s_idx].size(); i++)
	{
		int e_idx = dp[s_idx][i];
		res.push_back(s.substr(s_idx, e_idx-s_idx));

		if(e_idx == s.size())// we get a solution
		{
			string p;
			for(int k = 0; k < (res.size()-1); k++)
			{
				p.append(res[k]);
				p.append(" ");
			}
			p.append(res[res.size()-1]);
			result.push_back(p);
			res.pop_back();
			return;
		}

		WBS(dp, e_idx, s, res);
		res.pop_back();
	}
}

vector<string> wordBreak2(string s, vector<string>& wordDict) {
	unordered_set<string> Dict;
	vector<vector<int>> dp(s.size());
	vector<string> res;
	int min_size = INT_MAX, max_size = INT_MIN;

	for(vector<string>::iterator iter = wordDict.begin(); iter != wordDict.end(); iter++)
	{
		int size = (*iter).size();
		Dict.insert(*iter);
		min_size = min(min_size, size);
		max_size = max(max_size, size);
	}
	
	vector<bool> canbreak(s.size(), false);
	for(int i = min_size-1; i < s.size(); i++)
	{
		for(int j = i; j >= -1; j--)
		{
			if(j == -1 || canbreak[j])
			{
				if(Dict.find(s.substr(j+1, i-j)) != Dict.end())
				{
					canbreak[i] = true;
					if( j == -1) dp[0].push_back(i+1);
					else		 dp[j+1].push_back(i+1);
				}
			}

			if((i-j+1) > max_size) break;
		}
	}

	if(canbreak[s.size()-1])
	{
		vector<string> res;
		WBS(dp, 0, s, res);
	}

	return result;
}

/* 454. 4Sum II */
int fourSumCount(vector<int>& A, vector<int>& B, vector<int>& C, vector<int>& D) {
	unordered_map<int, int> AB;
	int i, j, res = 0;

	for(i = 0; i < A.size(); i++)
			for(j = 0; j < B.size(); j++)
			AB[A[i]+B[j]]++;

	for(i = 0; i < C.size(); i++){
		for(j = 0; j < D.size(); j++){
			unordered_map<int, int>::iterator it = AB.find(0-C[i]-D[j]);
			if(it != AB.end()) res += it->second;
		}
	}

	return res;
}

/* non-letcode, sum of the tree for all int value from root->leaf. */
typedef pair<TreeNode *, int> PT;

int sumNumbers(TreeNode *root) {
	stack<PT> st;
	st.push(make_pair(root,0));
	unsigned int val = 0, sum = 0;
	while(!st.empty())
	{
		PT p =  st.top();
		val =  p.first->val + p.second*10;
		st.pop();

		if(p.first->left == NULL && p.first->right == NULL)
		{
			sum += val;
			continue;
		}

		if(p.first->right) st.push(make_pair(p.first->right, val));
		if(p.first->left) st.push(make_pair(p.first->left, val));
	}

	return sum;
}

/* non-letcode, list all combination for k numbers in [1-> n]. */
vector<vector<int>> combine(int n, int k) {
	vector<vector<int>> res;
	vector<int> zero(1,0);
	
	for(int i = 1; i <= n; i++)
	{
		vector<int> p;
		p.push_back(i);
		res.push_back(p);
	}

	for(int j = 2; j <=k; j++)
	{
		int r, res_size =  res.size();
		for(r = 0; r < res_size; r++)
		{
			vector<int> p = res.front();
			for(int start_i = (p.back()+1); start_i <= n; start_i++)
			{
				vector<int> new_res = p;
				new_res.push_back(start_i);
				res.push_back(new_res);
			}
			res.erase(res.begin());
		}
	}

	return res;
}

/* 890. Find and Replace Pattern */
vector<int> wordPattern(string word)
{
	    vector<int> PAT;
		string map = "";

		for(int i = 0; i < word.size(); i++)
		{
			if(map.find(word[i]) == string::npos) map += word[i];
		}
		for(int i = 0; i < word.size(); i++)
		{
			PAT.push_back(map.find(word[i]));
		}

		return PAT;
}

vector<string> findAndReplacePattern(vector<string>& words, string pattern) {
        vector<int> PAT;
		vector<string> res;

		PAT = wordPattern(pattern);

		for(int i = 0; i < words.size(); i++)
		{
			vector<int> PATp = wordPattern(words[i]);
			if(PAT == PATp) res.push_back(words[i]);
		}

		return res;
}

bool Find(int target, vector<vector<int> > array) {
        bool ret = false;
		int rows, cols;
        rows = array.size();
        if(rows == 0) return ret;
        cols = array[0].size();
        if(cols == 0 || target > array[rows-1][cols-1] || target < array[0][0]) return ret;
        
        for(int i = 0; i < rows; i++)
        {
            if(target >= array[i][0] && target <= array[i][cols-1]){
                int  mid, li, ri;
                li = 0;
                ri = cols-1;
                mid = (li+ri)/2;
                while(li <= ri){
                    if(target > array[i][mid]) li = mid+1;
                    else if(target < array[i][mid]) ri = mid-1;
                    else return true;
                    mid = (li+ri)/2;
                }
            }
        }
        return ret;
}


TreeNode* reConstructBinaryTree(vector<int> pre,vector<int> vin) {
		if(pre.size() == 0 || vin.size() == 0)
			return NULL;

        TreeNode *root = new TreeNode(pre[0]);
		int root_idx = -1;
		for(int i = 0; i < vin.size(); i++)
		{
			if(vin[i] ==  pre[0]){
				root_idx = i; break;
			}
		}
        
		/* sub vin left tree */
		vector<int> vin_left;
		for(int i = 0; i < root_idx; i++)
		{
			vin_left.push_back(vin[i]);
		}
		/* sub vin right tree */
		vector<int> vin_right;
		for(int i = root_idx+1; i < vin.size(); i++)
			vin_right.push_back(vin[i]);

		/* Pre left vector */
		vector<int> pre_left;
		for(int i = 1; i <= root_idx; i++)
		{
			pre_left.push_back(pre[i]);
		}

		/* Pre right vector */
		vector<int> pre_right;
		for(int i = root_idx+1; i < pre.size(); i++)
		{
			pre_right.push_back(pre[i]);
		}

		root->left = reConstructBinaryTree(pre_left, vin_left);
		root->right = reConstructBinaryTree(pre_right, vin_right);

		return root;
}


class twostack_queue
{
public:
    void push(int node) {
        stack1.push(node);
    }

    int pop() {
        while(!stack1.empty())
        {
            stack2.push(stack1.top());
            stack1.pop();
        }
        stack2.pop();
        while(!stack2.empty())
        {
            stack1.push(stack2.top());
            stack2.pop();
        }
    }

private:
    stack<int> stack1;
    stack<int> stack2;
};

int minNumberInRotateArray(vector<int> rotateArray) {
        if(rotateArray.size() == 0) return 0;
        int res, pre;
        pre = rotateArray[0];
        res = rotateArray[0];
        for(int i = 1; i < rotateArray.size(); i++)
        {
            if(pre < rotateArray[i]){
                res = rotateArray[i]; break;
            }
            pre = rotateArray[i];
        }
        
        return res;
}

bool isSubTree(TreeNode* pRoot1, TreeNode* pRoot2)
{
	if(pRoot1->val != pRoot2->val) return false;
	queue<TreeNode*> p1, p2;
	/* BFS has sub result, for examle: [1,2,3,4,5,6]->[1,2,3]*/
	p1.push(pRoot1);
	p2.push(pRoot2);
	while(true)
	{
		if(p2.empty()) return true;
		if(p1.empty() && !p2.empty()) return false;

		TreeNode *node_p1 = p1.front();
		TreeNode *node_p2 = p2.front();
		if(node_p1->val == node_p2->val)
		{
			p1.pop();
			p2.pop();
			if(node_p1->left!= NULL) p1.push(node_p1->left);
			if(node_p1->right!= NULL)  p1.push(node_p1->right);
			if(node_p2->left!= NULL) p2.push(node_p2->left);
			if(node_p2->right!= NULL) p2.push(node_p2->right);
		}
		else
		{
			return false;
		}
	}
}

bool HasSubtree(TreeNode* pRoot1, TreeNode* pRoot2)
{
	if(pRoot2 == NULL) return false;
	if(pRoot1 == NULL) return false;

	return(isSubTree(pRoot1, pRoot2) || HasSubtree(pRoot1->left, pRoot2) || HasSubtree(pRoot1->right, pRoot2));
}

class Min_stack {
public:
    void push(int value) {
		st.push(value);
		min.push_back(value);
		sort(min.begin(), min.end());
    }
    void pop() {
        int val = st.top();
		st.pop();
		vector<int>::iterator it =  find(min.begin(), min.end(), val);
		min.erase(it);
    }
    int top() {
        return st.top();
    }
    int mins() {
        return min[0];
    }
private:
	stack<int> st;
	vector<int> min;
};

vector<vector<int>> PathRes;
void isFindPath(TreeNode* root,int expectNumber, vector<int> &res){
	if(root->left == NULL && root->right == NULL){
		res.push_back(root->val);
		if(root->val == expectNumber){
			PathRes.push_back(res);
		}
		return;
	}

	res.push_back(root->val);
	if(root->left != NULL){
		isFindPath(root->left, (expectNumber - root->val), res);
		res.pop_back();
	}
	if(root->right != NULL){
		isFindPath(root->right, (expectNumber - root->val), res);
		res.pop_back();
	}
}

vector<vector<int>> FindPath(TreeNode* root,int expectNumber) {
	vector<int> res;
	isFindPath(root, expectNumber, res);
	return PathRes;
}

TreeNode* Convert(TreeNode* pRootOfTree)
{
	/* midorder for the BST */
	if(pRootOfTree == NULL) return NULL;
	stack<TreeNode*> st;
	TreeNode *phead, *curp, *prep;

	phead = pRootOfTree;
	curp = pRootOfTree;
	prep = NULL;
	while((curp != NULL) || !st.empty())
	{
		while(curp != NULL)
		{
			st.push(curp);
			curp =  curp->left;
		}

		if(!st.empty()){
			curp = st.top();
			st.pop();
			curp->left = prep;
			if(prep != NULL) prep->right = curp;
			else			 phead =  curp;

			prep = curp;
			curp = curp->right;
		}
	}

	return phead;
}

int FindGreatestSumOfSubArray(vector<int> array) {
    int size = array.size(), maxs = INT_MIN;
	if(size == 0) return 0;

	vector<vector<int>> dp(size, vector<int>(size, 0));
	for(int i = 0; i < size; i++)
	{
		dp[i][i] = array[i];
		if(dp[i][i] > maxs) maxs = dp[i][i];
	}

	for(int i = 0; i < size; i++)
	{
		for(int j = i+1; j < size; j++)
		{
			dp[i][j] = dp[i][j-1] + array[j];
			if(dp[i][j] > maxs) maxs = dp[i][j];
		}
	}

	return maxs;
}

vector<int> printMatrix(vector<vector<int> > matrix) {
        int row = matrix.size();
        int col = matrix[0].size();
        vector<int> res;
        vector<vector<int>> pre_m = matrix;
		vector<vector<int>> new_m = matrix;
        
        while(row != 0 && col != 0)
        {
            for(int i = 0; i < col; i++)
                res.push_back(new_m[0][i]);
            
            /* Reverse the remain vectors in mirror */
			new_m.clear();
			for(int i = 0; i < col; i++)
			{
				vector<int> temp;
				for(int j = 1; j < row; j++)
				{
					temp.push_back(pre_m[j][i]);
				}
				new_m.insert(new_m.begin(), temp);
			}
			row = col;
			col = row - 1;
			pre_m = new_m;
        }

		return res;
}

TreeNode* KthNode(TreeNode* pRoot, int k)
{
	stack<TreeNode*> st;
	TreeNode *pNode = pRoot;

	while(pNode != NULL || !st.empty())
	{
		if(pNode != NULL){
			st.push(pNode);
			pNode = pNode->left;
		}
		else{
			pNode = st.top();
			st.pop();
			cout << pNode->val << endl;
			k--;
			if(k == 0) return pNode;
			pNode = pNode->right;
		}
	}

	return pNode;
}


void qsort(int *num, int low, int high)
{
	if(low >= high) return;
	int key = num[low], keyi = low;
	int i = low, j = high;
	
	while(i < j)
	{
		if(num[i] < num[j]){
			j--;
		}
		else
		{
			swap(num[i],num[j]);
			keyi = j;
			while(i < j)
			{
				if(num[i] <= num[j]){
					i++;
				}
				else
				{
					swap(num[i], num[j]);
					keyi = i;
					break;
				}
			}
		}

	}

	qsort(num, low, keyi-1);
	qsort(num, keyi+1, high);
}

int GetUglyNumber_Solution(int index) {
	vector<int> res;
	res.push_back(1);
	int s1 = 0;
	int s2 = 0;
	int s3 = 0;

	for(int i = 1; i < index; i++)
	{
		int ret = min(res[s1]*2, min(res[s2]*3, res[s3]*5));
		if(res[s1]*2 == ret) s1++;
		if(res[s2]*3 == ret) s2++;
		if(res[s3]*5 == ret) s3++;
		res.push_back(ret);
	}

	return res.back();
}

vector<int> maxInWindows(const vector<int>& num, unsigned int size)
{
        list<int> win;
        vector<int> res;
        int c_max = INT_MIN, i = 0;
        if(num.size() < size) return res;
        
        for(; i < size; i++){
            win.push_back(num[i]);
            if(num[i] > c_max) c_max = num[i];
        }
        res.push_back(c_max);
        
        for(; i < num.size(); i++){
            int front = win.front();
            win.pop_front();
            win.push_back(num[i]);
            if(front == c_max){
                c_max = INT_MIN;
                for(list<int>::iterator it = win.begin(); it != win.end(); it++){
                    if(*it > c_max) c_max = *it;
                }
            }else{
                if(num[i] > c_max) c_max = num[i];
            }
            res.push_back(c_max);
        }
        
        return res;
}

bool path_dfs(char* matrix, int rows, int cols, int i, int j, char *str, int idx, vector<bool> &visit)
{
	int result = false, pos;
    //reach the end of str, bingo
	if(idx == strlen(str)) return true;
	//out of index
	if(i < 0 || i >= rows || j < 0 || j >= cols) return false;
	pos = i*cols + j;
	if(pos < 0 || pos > strlen(matrix)) return false;

	if((matrix[pos] == str[idx]) && (visit[pos] == false)){
		//find 4 directions to see if any match
		visit[pos] = true;
		result = path_dfs(matrix, rows, cols, i-1, j, str, idx+1, visit) || 
				 path_dfs(matrix, rows, cols, i, j-1, str, idx+1, visit) ||
				 path_dfs(matrix, rows, cols, i, j+1, str, idx+1, visit) ||
				 path_dfs(matrix, rows, cols, i+1, j, str, idx+1, visit);
		visit[pos] = false;
	}
	return result;
}

bool hasPath(char* matrix, int rows, int cols, char* str)
{
        if(matrix == NULL || str == NULL) return false;
        vector<bool> visit(strlen(matrix), false);
        for(int i = 0; i < rows; i++){
            for(int j = 0; j < cols; j++){
                if(matrix[i*cols+j] == str[0]){
					if(path_dfs(matrix, rows, cols, i, j, str, 0, visit)) return true;
				}
            }
        }

		return false;
}

/* 1028. Recover a Tree From Preorder Traversal */
TreeNode *NodePreorder(string S, int depth)
{
	if(S.empty()) return NULL;

	int index = 0, dep = 0;
	int first_idx = std::string::npos, second_idx = std::string::npos;

	while(index < S.size() && S[index] != '-') index++;
	TreeNode *pNode = new TreeNode(stoi(S.substr(0, index)));

	while(index < S.size() && S[index] == '-') {index++; dep++;}
	if(dep == depth) first_idx = index;

	if(first_idx == std::string::npos){
		//leaf node
		return pNode;
	}

	while(index < S.size()){
		dep = 0;
		while(index < S.size() && S[index] == '-') {dep++; index++;}
		if(dep == depth) {second_idx = index; break;}
		index++;
	}

	if(second_idx == std::string::npos){
		//no right son
		pNode->left = NodePreorder(S.substr(first_idx), depth+1);
		pNode->right = NULL;
	}
	else{
		//both left and right son exist
		int l_len = second_idx - first_idx - depth; //skip the '-' prefix
		pNode->left = NodePreorder(S.substr(first_idx, l_len), depth+1);
		pNode->right = NodePreorder(S.substr(second_idx), depth+1);
	}

	return pNode;
}

TreeNode* recoverFromPreorder(string S) {
	if(S.empty()) return NULL;
	return NodePreorder(S, 1);
}


int main(int argc, char* argv[])
{
	int a[] = {1,2,3,4};
	int a1[] = {5,6,7,8};
	int a2[] = {9,10,11,12};
	int a3[] = {13,14,15,16};
	vector<vector<int>> matrix;
	vector<int> ma(a, a+4);
	matrix.push_back(ma);
	vector<int> ma1(a1, a1+4);
	matrix.push_back(ma1);
	vector<int> ma2(a2, a2+4);
	matrix.push_back(ma2);
	vector<int> ma3(a3, a3+4);
	matrix.push_back(ma3);

	printMatrix(matrix);

	int in[] = {10, 5, 12, 4, 7};
	TreeNode root(in[0]);
	TreeNode root1(in[1]);
	TreeNode root2(in[2]);
	TreeNode root3(in[3]);
	TreeNode root4(in[4]);
	root.left = &root1;
	root.right = &root2;
	root1.left = &root3;
	root1.right = &root4;
	FindPath(&root, 22);

	int in2[] = {3, 10, 60, 9, 10, 54, 3, 27, 126, 12, 89, 0};
	qsort(in2, 0, 11);

	int in3[] = {2,3,4,2,6,2,5,1};
	vector<int> num(in3, in3+8);
	maxInWindows(num, 10);

	GetUglyNumber_Solution(3);

	priority_queue<int, vector<int>, less<int>> pq;

	string S = "1-2--3---4-5--6---7";
	TreeNode *res  = recoverFromPreorder(S);

    //printf("result: %d\n",isMatch2("aasdfasdfasdfasdfas", "aasdf.*asdf.*asdf.*asdf.*s"));
	system("pause");

	return 0;
}

