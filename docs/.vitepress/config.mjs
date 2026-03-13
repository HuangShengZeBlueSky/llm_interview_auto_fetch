import { defineConfig } from 'vitepress'

export default defineConfig({
  title: "LLM 知识库",
  description: "面经、论文、课程三位一体的自动化知识库",
  base: "/llm_interview_auto_fetch/",
  themeConfig: {
    nav: [
      { text: '首页', link: '/' },
      { text: '面经', link: '/reports/interviews/' },
      { text: '论文', link: '/reports/papers/' },
      { text: '课程', link: '/reports/courses/' },
      { text: '洞察', link: '/reports/insights/' },
      { text: '仓库结构', link: '/repo-structure' }
    ],
    sidebar: {
      '/reports/': [
      {
            "text": "专题导航",
            "collapsed": false,
            "items": [
                  {
                        "text": "知识库总览",
                        "link": "/reports/"
                  },
                  {
                        "text": "面经板块",
                        "link": "/reports/interviews/"
                  },
                  {
                        "text": "论文板块",
                        "link": "/reports/papers/"
                  },
                  {
                        "text": "课程板块",
                        "link": "/reports/courses/"
                  },
                  {
                        "text": "洞察板块",
                        "link": "/reports/insights/"
                  }
            ]
      },
      {
            "text": "面经题库",
            "collapsed": false,
            "items": [
                  {
                        "text": "专题题库",
                        "collapsed": true,
                        "items": [
                              {
                                    "text": "AI算法高频题",
                                    "collapsed": true,
                                    "items": [
                                          {
                                                "text": "大厂AI算法高频题按模块分类",
                                                "link": "/reports/专题题库/AI算法高频题/20260313_大厂AI算法高频题按模块分类"
                                          },
                                          {
                                                "text": "大厂AI算法高频题总表",
                                                "link": "/reports/专题题库/AI算法高频题/20260313_大厂AI算法高频题总表"
                                          }
                                    ]
                              }
                        ]
                  },
                  {
                        "text": "复试准备",
                        "collapsed": true,
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
                        "collapsed": true,
                        "items": [
                              {
                                    "text": "LLM基础",
                                    "collapsed": true,
                                    "items": [
                                          {
                                                "text": "算法面经：字节豆包大模型11.13_1_AI实战领航员_来自小红书网页版",
                                                "link": "/reports/字节跳动/LLM基础/20260312_112054_算法面经-字节豆包大模型11.13_1_AI实战领航员_来自小红书网页版"
                                          },
                                          {
                                                "text": "算法面经：字节豆包大模型11.13_3_AI实战领航员_来自小红书网页版",
                                                "link": "/reports/字节跳动/LLM基础/20260312_101616_算法面经-字节豆包大模型11.13_3_AI实战领航员_来自小红书网页版"
                                          },
                                          {
                                                "text": "算法面经：字节豆包大模型11.13_2_AI实战领航员_来自小红书网页版",
                                                "link": "/reports/字节跳动/LLM基础/20260312_101445_算法面经-字节豆包大模型11.13_2_AI实战领航员_来自小红书网页版"
                                          },
                                          {
                                                "text": "算法面经：字节豆包大模型11.13_1_AI实战领航员_来自小红书网页版",
                                                "link": "/reports/字节跳动/LLM基础/20260312_101317_算法面经-字节豆包大模型11.13_1_AI实战领航员_来自小红书网页版"
                                          }
                                    ]
                              }
                        ]
                  },
                  {
                        "text": "通用_未知",
                        "collapsed": true,
                        "items": [
                              {
                                    "text": "LLM基础",
                                    "collapsed": true,
                                    "items": [
                                          {
                                                "text": "test_classification",
                                                "link": "/reports/通用_未知/LLM基础/20260312_110244_test_classification"
                                          },
                                          {
                                                "text": "2",
                                                "link": "/reports/通用_未知/LLM基础/20260312_101213_2"
                                          },
                                          {
                                                "text": "1",
                                                "link": "/reports/通用_未知/LLM基础/20260312_100947_1"
                                          }
                                    ]
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
                        "text": "高分论文索引",
                        "collapsed": true,
                        "items": [
                              {
                                    "text": "顶会高分论文总览",
                                    "link": "/reports/论文精读/高分论文索引/20260313_顶会高分论文总览"
                              }
                        ]
                  },
                  {
                        "text": "AI for Science",
                        "collapsed": true,
                        "items": [
                              {
                                    "text": "AI for Science",
                                    "link": "/reports/论文精读/AI-for-Science/20260313_顶会高分论文_AI-for-Science"
                              }
                        ]
                  },
                  {
                        "text": "Agent & Planning",
                        "collapsed": true,
                        "items": [
                              {
                                    "text": "Paper_arxiv_20260312125327_COMIC Age",
                                    "link": "/reports/论文精读/Agent-and-Planning/20260312_125620_Paper_arxiv_20260312125327_COMIC-Age"
                              }
                        ]
                  },
                  {
                        "text": "Agents & Tool Use",
                        "collapsed": true,
                        "items": [
                              {
                                    "text": "Agents & Tool Use",
                                    "link": "/reports/论文精读/Agents-and-Tool-Use/20260313_顶会高分论文_Agents-and-Tool-Use"
                              }
                        ]
                  },
                  {
                        "text": "Alignment & Post-training",
                        "collapsed": true,
                        "items": [
                              {
                                    "text": "Alignment & Post-training",
                                    "link": "/reports/论文精读/Alignment-and-Post-training/20260313_顶会高分论文_Alignment-and-Post-training"
                              }
                        ]
                  },
                  {
                        "text": "Efficiency & Systems",
                        "collapsed": true,
                        "items": [
                              {
                                    "text": "Efficiency & Systems",
                                    "link": "/reports/论文精读/Efficiency-and-Systems/20260313_顶会高分论文_Efficiency-and-Systems"
                              }
                        ]
                  },
                  {
                        "text": "Evaluation & Benchmarks",
                        "collapsed": true,
                        "items": [
                              {
                                    "text": "Evaluation & Benchmarks",
                                    "link": "/reports/论文精读/Evaluation-and-Benchmarks/20260313_顶会高分论文_Evaluation-and-Benchmarks"
                              }
                        ]
                  },
                  {
                        "text": "Multimodal Models",
                        "collapsed": true,
                        "items": [
                              {
                                    "text": "Multimodal Models",
                                    "link": "/reports/论文精读/Multimodal-Models/20260313_顶会高分论文_Multimodal-Models"
                              },
                              {
                                    "text": "Paper_arxiv_20260312125327_V2M-Zero ",
                                    "link": "/reports/论文精读/Multimodal-Models/20260312_165448_Paper_arxiv_20260312125327_V2M-Zero"
                              }
                        ]
                  },
                  {
                        "text": "Parameter-Efficient Fine-Tuning",
                        "collapsed": true,
                        "items": [
                              {
                                    "text": "Paper_LoRA_Paper_Abstract",
                                    "link": "/reports/论文精读/Parameter-Efficient-Fine-Tuning/20260312_165521_Paper_LoRA_Paper_Abstract"
                              }
                        ]
                  },
                  {
                        "text": "RAG & Knowledge Editing",
                        "collapsed": true,
                        "items": [
                              {
                                    "text": "RAG & Knowledge Editing",
                                    "link": "/reports/论文精读/RAG-and-Knowledge-Editing/20260313_顶会高分论文_RAG-and-Knowledge-Editing"
                              }
                        ]
                  },
                  {
                        "text": "Reasoning & CoT",
                        "collapsed": true,
                        "items": [
                              {
                                    "text": "Reasoning & CoT",
                                    "link": "/reports/论文精读/Reasoning-and-CoT/20260313_顶会高分论文_Reasoning-and-CoT"
                              }
                        ]
                  },
                  {
                        "text": "World Models & Robotics",
                        "collapsed": true,
                        "items": [
                              {
                                    "text": "World Models & Robotics",
                                    "link": "/reports/论文精读/World-Models-and-Robotics/20260313_顶会高分论文_World-Models-and-Robotics"
                              }
                        ]
                  },
                  {
                        "text": "其他",
                        "collapsed": true,
                        "items": [
                              {
                                    "text": "Paper_arxiv_20260312125327_Neural Fi",
                                    "link": "/reports/论文精读/其他/20260312_165409_Paper_arxiv_20260312125327_Neural-Fi"
                              },
                              {
                                    "text": "Paper_arxiv_20260312125327_LiTo Surf",
                                    "link": "/reports/论文精读/其他/20260312_165327_Paper_arxiv_20260312125327_LiTo-Surf"
                              },
                              {
                                    "text": "Paper_arxiv_20260312125327_Instructi",
                                    "link": "/reports/论文精读/其他/20260312_125700_Paper_arxiv_20260312125327_Instructi"
                              }
                        ]
                  }
            ]
      },
      {
            "text": "课程笔记",
            "collapsed": false,
            "items": [
                  {
                        "text": "CS224N (NLP基础)",
                        "collapsed": true,
                        "items": [
                              {
                                    "text": "Note_CS224N_Lecture1_Notes",
                                    "link": "/reports/体系化课程/CS224N-(NLP基础)/20260312_165246_Note_CS224N_Lecture1_Notes"
                              }
                        ]
                  }
            ]
      },
      {
            "text": "行业洞察",
            "collapsed": false,
            "items": [
                  {
                        "text": "最新风向标",
                        "collapsed": true,
                        "items": [
                              {
                                    "text": "大盘洞察",
                                    "link": "/reports/00_行业洞察/最新风向标/20260312_1124_大盘洞察"
                              },
                              {
                                    "text": "大盘洞察",
                                    "link": "/reports/00_行业洞察/最新风向标/20260312_1108_大盘洞察"
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
