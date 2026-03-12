import { defineConfig } from 'vitepress'

export default defineConfig({
  title: "LLM 面试题库",
  description: "AI 自动提取与全量解析",
  themeConfig: {
    nav: [
      { text: '🏠 首页', link: '/' },
      { text: '📚 题库大全', link: '/reports/' }
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
            "text": "字节跳动",
            "collapsed": false,
            "items": [
                  {
                        "text": "LLM基础",
                        "collapsed": true,
                        "items": [
                              {
                                    "text": "算法面经：字节豆包大模型11.13_1_AI实战领航员_来自小红书网页版",
                                    "link": "/reports/字节跳动/LLM基础/20260312_112054_算法面经：字节豆包大模型11.13_1_AI实战领航员_来自小红书网页版"
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