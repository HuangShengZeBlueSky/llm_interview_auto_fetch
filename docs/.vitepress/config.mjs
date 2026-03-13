import { defineConfig } from 'vitepress'

export default defineConfig({
  title: "LLM 面试题库",
  description: "AI 自动提取与全量解析",
  base: "/llm_interview_auto_fetch/",
  themeConfig: {
    nav: [
      { text: '🏠 首页', link: '/' },
      { text: '📚 题库大全', link: '/reports/' },
      { text: '🧭 仓库结构', link: '/repo-structure' }
    ],
    sidebar: {
      '/reports/': [
      {
            "text": "00/行业洞察",
            "collapsed": false,
            "items": [
                  {
                        "text": "最新风向标",
                        "collapsed": true,
                        "items": [
                              {
                                    "text": "大盘洞察",
                                    "link": "/reports/00_行业洞察/最新风向标/20260312_1108_大盘洞察"
                              },
                              {
                                    "text": "大盘洞察",
                                    "link": "/reports/00_行业洞察/最新风向标/20260312_1124_大盘洞察"
                              }
                        ]
                  }
            ]
      },
      {
            "text": "体系化课程",
            "collapsed": false,
            "items": [
                  {
                        "text": "CS224N (NLP基础)",
                        "collapsed": true,
                        "items": [
                              {
                                    "text": "Note_CS224N_Lecture1_Notes",
                                    "link": "/reports/体系化课程/CS224N (NLP基础)/20260312_165246_Note_CS224N_Lecture1_Notes"
                              }
                        ]
                  }
            ]
      },
      {
            "text": "复试准备",
            "collapsed": false,
            "items": [
                  {
                        "text": "RAG与多智能体",
                        "collapsed": true,
                        "items": [
                              {
                                    "text": "RAG优化与多智能体方案",
                                    "link": "/reports/复试准备/RAG与多智能体/20260312_复试项目深挖_RAG优化与多智能体方案"
                              }
                        ]
                  }
            ]
      },
      {
            "text": "字节跳动",
            "collapsed": false,
            "items": [
                  {
                        "text": "LLM基础",
                        "collapsed": true,
                        "items": [
                              {
                                    "text": "算法面经：字节豆包大模型11.13_1_AI实战领航员_来自小红书网页版",
                                    "link": "/reports/字节跳动/LLM基础/20260312_101317_算法面经：字节豆包大模型11.13_1_AI实战领航员_来自小红书网页版"
                              },
                              {
                                    "text": "算法面经：字节豆包大模型11.13_2_AI实战领航员_来自小红书网页版",
                                    "link": "/reports/字节跳动/LLM基础/20260312_101445_算法面经：字节豆包大模型11.13_2_AI实战领航员_来自小红书网页版"
                              },
                              {
                                    "text": "算法面经：字节豆包大模型11.13_3_AI实战领航员_来自小红书网页版",
                                    "link": "/reports/字节跳动/LLM基础/20260312_101616_算法面经：字节豆包大模型11.13_3_AI实战领航员_来自小红书网页版"
                              },
                              {
                                    "text": "算法面经：字节豆包大模型11.13_1_AI实战领航员_来自小红书网页版",
                                    "link": "/reports/字节跳动/LLM基础/20260312_112054_算法面经：字节豆包大模型11.13_1_AI实战领航员_来自小红书网页版"
                              }
                        ]
                  }
            ]
      },
      {
            "text": "论文精读",
            "collapsed": false,
            "items": [
                  {
                        "text": "Agent & Planning",
                        "collapsed": true,
                        "items": [
                              {
                                    "text": "Paper_arxiv_20260312125327_COMIC Age",
                                    "link": "/reports/论文精读/Agent & Planning/20260312_125620_Paper_arxiv_20260312125327_COMIC Age"
                              }
                        ]
                  },
                  {
                        "text": "Multimodal Models",
                        "collapsed": true,
                        "items": [
                              {
                                    "text": "Paper_arxiv_20260312125327_V2M-Zero ",
                                    "link": "/reports/论文精读/Multimodal Models/20260312_165448_Paper_arxiv_20260312125327_V2M-Zero "
                              }
                        ]
                  },
                  {
                        "text": "Parameter-Efficient Fine-Tuning",
                        "collapsed": true,
                        "items": [
                              {
                                    "text": "Paper_LoRA_Paper_Abstract",
                                    "link": "/reports/论文精读/Parameter-Efficient Fine-Tuning/20260312_165521_Paper_LoRA_Paper_Abstract"
                              }
                        ]
                  },
                  {
                        "text": "其他",
                        "collapsed": true,
                        "items": [
                              {
                                    "text": "Paper_arxiv_20260312125327_Instructi",
                                    "link": "/reports/论文精读/其他/20260312_125700_Paper_arxiv_20260312125327_Instructi"
                              },
                              {
                                    "text": "Paper_arxiv_20260312125327_LiTo Surf",
                                    "link": "/reports/论文精读/其他/20260312_165327_Paper_arxiv_20260312125327_LiTo Surf"
                              },
                              {
                                    "text": "Paper_arxiv_20260312125327_Neural Fi",
                                    "link": "/reports/论文精读/其他/20260312_165409_Paper_arxiv_20260312125327_Neural Fi"
                              }
                        ]
                  }
            ]
      },
      {
            "text": "通用/未知",
            "collapsed": false,
            "items": [
                  {
                        "text": "LLM基础",
                        "collapsed": true,
                        "items": [
                              {
                                    "text": "1",
                                    "link": "/reports/通用_未知/LLM基础/20260312_100947_1"
                              },
                              {
                                    "text": "2",
                                    "link": "/reports/通用_未知/LLM基础/20260312_101213_2"
                              },
                              {
                                    "text": "test_classification",
                                    "link": "/reports/通用_未知/LLM基础/20260312_110244_test_classification"
                              }
                        ]
                  }
            ]
      }
]
    },
    socialLinks: [
      { icon: 'github', link: 'https://github.com/HuangShengZeBlueSky/llm_interview_auto_fetch' }
    ],
    search: {
      provider: 'local'
    }
  }
})