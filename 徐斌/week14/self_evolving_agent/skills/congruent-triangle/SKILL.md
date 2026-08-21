---
name: congruent-triangle
description: 匹配「全等三角形的判定（SSS）」类习题：题干通过三边相等（含中点、等长线段、等腰性质导出）证全等，进而推垂直平分线、角相等或线段关系。
version: 2
---

# 全等三角形的判定（SSS）

## 输出名称（必须一字不差）
全等三角形的判定（SSS）

## 触发特征
- 含三边相等表述：如“AB=AC”“AD=AE”“BD=BC”“E是AB中点”“FM=DM”
- 明确SSS句式：“求证△ADE≌△BFE”“由△ABD≌△ACD可得”
- 图形隐含三组对应边等（如等腰+中点→AC=AB, AD=AD, CD=BD）

## 反例（不要匹配）
- 含“ASA”“AAS”“SAS”“HL”等其他判定法字眼
- 仅两组边等+夹角，未提/无法推出第三边等
