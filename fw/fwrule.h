#pragma once

constexpr size_t kDefaultFrames = 49;
constexpr size_t kDefaultInterpolation = 15;
constexpr float kDefaultScaleRate = 0.025f;

// 为了方便规则的复用，本文件定义了所有的解析规则

/* ==========================
 * Normal Firework
 * ==========================
 */

#define NORMAL_RULE_GROUP(_x)             \
	BeginGroup(1, 3);                     \
	AddColorGroup("初始颜色" + _x);       \
	EndGroup();                           \
	BeginGroup(1, 1);                     \
	AddScalarGroup("初始尺寸" + _x);      \
	EndGroup();                           \
	BeginGroup(1, 1);                     \
	AddScalarGroup("X方向加速度" + _x);   \
	EndGroup();                           \
	BeginGroup(1, 1);                     \
	AddScalarGroup("Y方向加速度" + _x);   \
	EndGroup();                           \
	BeginGroup(1, 1);                     \
	AddScalarGroup("离心速度" + _x);      \
	EndGroup();                           \
	BeginGroup(1, 1);                     \
	AddScalarGroup("内环尺寸" + _x);      \
	EndGroup();                           \
	BeginGroup(1, 1);                     \
	AddScalarGroup("内环色彩增强" + _x);  \
	EndGroup();              

#define NORMAL_RULE_VALUE(_x)             \
	AddValue("颜色衰减率" + _x);          \
	AddValue("尺寸衰减率" + _x);          \
	AddVec3("初始位置" + _x);             \
	AddValue("横截面粒子数量" + _x);      \
	AddValue("随机比率" + _x);            \
	AddValue("寿命" + _x);

constexpr size_t kDefaultNormalArgs = kDefaultFrames * 9 + 8;
#define kDynamicNormalArgs (nFrames_ * 9 + 8)

/* ==========================
 * Circle Firework
 * ==========================
 */

#define CIRCLE_RULE_GROUP(_x) NORMAL_RULE_GROUP(_x)
#define CIRCLE_RULE_VALUE(_x)            \
	NORMAL_RULE_VALUE(_x)                \
	AddVec3("爆炸平面法线" + _x);        \
	AddValue("相对于水平面的角度" + _x);

constexpr size_t kDefaultCircleArgs = kDefaultNormalArgs + 4;
#define kDynamicCircleArgs (kDynamicNormalArgs + 4)

/* ==========================
 * Strafe Firework
 * ==========================
 */

#define STRAFE_RULE_GROUP(_x) NORMAL_RULE_GROUP(_x)
#define STRAFE_RULE_VALUE(_x) NORMAL_RULE_VALUE(_x)

constexpr size_t kDefaultStrafeArgs = kDefaultNormalArgs;
#define kDynamicStrafeArgs kDynamicNormalArgs

/* ==========================
 * Twinkle Firework
 * ==========================
 */

#define TWINKLE_RULE_GROUP(_x)              \
	NORMAL_RULE_GROUP(_x)                   \
	BeginGroup(1, 1);                       \
	AddScalarGroup("可见粒子束比例" + _x);  \
	EndGroup();
#define TWINKLE_RULE_VALUE(_x) NORMAL_RULE_VALUE(_x)

constexpr size_t kDefaultTwinkleArgs = kDefaultNormalArgs + kDefaultFrames;
#define kDynamicTwinkleArgs (kDynamicNormalArgs + kDefaultFrames)

/* ==========================
 * Multi-explosion Firework
 * ==========================
 */

#define MULTI_EXPLOSION_RULE_GROUP(_x) NORMAL_RULE_GROUP(_x)
#define MULTI_EXPLOSION_RULE_VALUE(_x)  \
	NORMAL_RULE_VALUE(_x)               \
	AddValue("二次爆炸时间" + _x);      \
	AddValue("二次爆炸比率" + _x);		\
	AddValue("子烟花横截面粒子数" + _x);

constexpr size_t kDefaultMultiExplosionArgs = kDefaultNormalArgs + 2;
#define kDynamicMultiExplosionArgs (kDynamicNormalArgs + 2)