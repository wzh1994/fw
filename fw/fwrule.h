#pragma once

constexpr size_t kDefaultFrames = 49;
constexpr size_t kDefaultInterpolation = 15;
constexpr float kDefaultScaleRate = 0.025f;

// Ϊ�˷������ĸ��ã����ļ����������еĽ�������

/* ==========================
 * Normal Firework
 * ==========================
 */

#define NORMAL_RULE_GROUP(_x)             \
	BeginGroup(1, 3);                     \
	AddColorGroup("��ʼ��ɫ" + _x);       \
	EndGroup();                           \
	BeginGroup(1, 1);                     \
	AddScalarGroup("��ʼ�ߴ�" + _x);      \
	EndGroup();                           \
	BeginGroup(1, 1);                     \
	AddScalarGroup("X������ٶ�" + _x);   \
	EndGroup();                           \
	BeginGroup(1, 1);                     \
	AddScalarGroup("Y������ٶ�" + _x);   \
	EndGroup();                           \
	BeginGroup(1, 1);                     \
	AddScalarGroup("�����ٶ�" + _x);      \
	EndGroup();                           \
	BeginGroup(1, 1);                     \
	AddScalarGroup("�ڻ��ߴ�" + _x);      \
	EndGroup();                           \
	BeginGroup(1, 1);                     \
	AddScalarGroup("�ڻ�ɫ����ǿ" + _x);  \
	EndGroup();              

#define NORMAL_RULE_VALUE(_x)             \
	AddValue("��ɫ˥����" + _x);          \
	AddValue("�ߴ�˥����" + _x);          \
	AddVec3("��ʼλ��" + _x);             \
	AddValue("�������������" + _x);      \
	AddValue("�������" + _x);            \
	AddValue("����" + _x);

constexpr size_t kDefaultNormalArgs = kDefaultFrames * 9 + 8;
#define kDynamicNormalArgs (nFrames_ * 9 + 8)

/* ==========================
 * Circle Firework
 * ==========================
 */

#define CIRCLE_RULE_GROUP(_x) NORMAL_RULE_GROUP(_x)
#define CIRCLE_RULE_VALUE(_x)            \
	NORMAL_RULE_VALUE(_x)                \
	AddVec3("��ըƽ�淨��" + _x);        \
	AddValue("�����ˮƽ��ĽǶ�" + _x);

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
	AddScalarGroup("�ɼ�����������" + _x);  \
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
	AddValue("���α�ըʱ��" + _x);      \
	AddValue("���α�ը����" + _x);		\
	AddValue("���̻������������" + _x);

constexpr size_t kDefaultMultiExplosionArgs = kDefaultNormalArgs + 2;
#define kDynamicMultiExplosionArgs (kDynamicNormalArgs + 2)