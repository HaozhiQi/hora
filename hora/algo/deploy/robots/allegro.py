# --------------------------------------------------------
# In-Hand Object Rotation via Rapid Motor Adaptation
# https://arxiv.org/abs/2210.04887
# Copyright (c) 2022 Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Based on:
# https://github.com/felixduvallet/allegro-hand-ros/blob/master/allegro_hand/src/allegro_hand/liballegro.py
# --------------------------------------------------------

import rospy
from std_msgs.msg import String
from std_msgs.msg import Float32
from sensor_msgs.msg import JointState


class Allegro(object):

    def __init__(self, hand_topic_prefix='allegroHand_0', num_joints=16):
        """ Simple python interface to the Allegro Hand.

        The AllegroClient is a simple python interface to an allegro
        robot hand.  It enables you to command the hand directly through
        python library calls (joint positions, joint torques, or 'named'
        grasps).

        The constructors sets up publishers and subscribes to the joint states
        topic for the hand.

        Note on hand topic names: The default topic (allegroHand/foo) can be
        remapped to a different topic prefix (allegroHand_0/foo) in one of two
        ways:
          1. pass in allegroHand_0 as the hand_topic_prefix
          2. remap *each* topic on the command line
             (allegroHand/joint_cmd:=allegroHand_0/joint_cmd)
        The first method is probably easier.

        :param hand_topic_prefix: The prefix to use for *all* hand
        topics (publishing & subscribing).

        :param num_joints: Number of expected joints, used when
        commanding joint positions.

        """

        # Topics (that can be remapped) for named graps
        # (ready/envelop/grasp/etc.), joint commands (position and
        # velocity), joint state (subscribing), and envelop torque. Note that
        # we can change the hand topic prefix (for example, to allegroHand_0)
        # instead of remapping it at the command line.
        hand_topic_prefix=hand_topic_prefix.rstrip('/')
        topic_grasp_command = '{}/lib_cmd'.format(hand_topic_prefix)
        topic_joint_command = '{}/joint_cmd'.format(hand_topic_prefix)
        topic_joint_state = '{}/joint_states'.format(hand_topic_prefix)
        topic_envelop_torque = '{}/envelop_torque'.format(hand_topic_prefix)

        # Publishers for above topics.
        self.pub_grasp = rospy.Publisher(
            topic_grasp_command, String, queue_size=10)
        self.pub_joint = rospy.Publisher(
            topic_joint_command, JointState, queue_size=10)
        self.pub_envelop_torque = rospy.Publisher(
            topic_envelop_torque, Float32, queue_size=1)
        rospy.Subscriber(topic_joint_state, JointState,
                         self._joint_state_callback)
        self._joint_state = None

        self._num_joints = num_joints

        rospy.loginfo('Allegro Client start with hand topic: {}'.format(
            hand_topic_prefix))

        # "Named" grasps are those provided by the bhand library. These can be
        # commanded directly and the hand will execute them. The keys are more
        # human-friendly names, the values are the expected names from the
        # allegro controller side. Multiple strings mapping to the same value
        # are allowed.
        self._named_grasps_mappings = {
            'home': 'home',
            'ready': 'ready',
            'three_finger_grasp': 'grasp_3',
            'three finger grasp': 'grasp_3',
            'four_finger_grasp': 'grasp_4',
            'four finger grasp': 'grasp_4',
            'index_pinch': 'pinch_it',
            'index pinch': 'pinch_it',
            'middle_pinch': 'pinch_mt',
            'middle pinch': 'pinch_mt',
            'envelop': 'envelop',
            'off': 'off',
            'gravity_compensation': 'gravcomp',
            'gravity compensation': 'gravcomp',
            'gravity': 'gravcomp'
        }

    def disconnect(self):
        """
        Disconnect the allegro client from the hand by sending the 'off'
        command. This is principally a convenience binding.

        Note that we don't actually 'disconnect', so you could technically
        continue sending other commands after this.
        """
        self.command_hand_configuration('off')

    def _joint_state_callback(self, data):
        self._joint_state = data

    def command_joint_position(self, desired_pose):
        """
        Command a specific desired hand pose.

        The desired pose must be the correct dimensionality (self._num_joints).
        Only the pose is commanded, and **no bound-checking happens here**:
        any commanded pose must be valid or Bad Things May Happen. (Generally,
        values between 0.0 and 1.5 are fine, but use this at your own risk.)

        :param desired_pose: The desired joint configurations.
        :return: True if pose is published, False otherwise.
        """

        # Check that the desired pose can have len() applied to it, and that
        # the number of dimensions is the same as the number of hand joints.
        if (not hasattr(desired_pose, '__len__') or
                len(desired_pose) != self._num_joints):
            rospy.logwarn('Desired pose must be a {}-d array: got {}.'
                          .format(self._num_joints, desired_pose))
            return False

        msg = JointState()  # Create and publish
        try:
            msg.position = desired_pose
            self.pub_joint.publish(msg)
            rospy.logdebug('Published desired pose.')
            return True
        except rospy.exceptions.ROSSerializationException:
            rospy.logwarn('Incorrect type for desired pose: {}.'.format(
                desired_pose))
            return False

    def command_joint_torques(self, desired_torques):
        """
        Command a desired torque for each joint.

        The desired torque must be the correct dimensionality
        (self._num_joints). Similarly to poses, we do not sanity-check
        the inputs. As a rule of thumb, values between +- 0.5 are fine.

        :param desired_torques: The desired joint torques.
        :return: True if message is published, False otherwise.
        """

        # Check that the desired torque vector can have len() applied to it,
        # and that the number of dimensions is the same as the number of
        # joints. This prevents passing singletons or incorrectly-shaped lists
        # to the message creation (which does no checking).
        if (not hasattr(desired_torques, '__len__') or
                len(desired_torques) != self._num_joints):
            rospy.logwarn('Desired torques must be a {}-d array: got {}.'
                          .format(self._num_joints, desired_torques))
            return False

        msg = JointState()  # Create and publish
        try:
            msg.effort = desired_torques
            self.pub_joint.publish(msg)
            rospy.logdebug('Published desired torques.')
            return True
        except rospy.exceptions.ROSSerializationException:
            rospy.logwarn('Incorrect type for desired torques: {}.'.format(
                desired_torques))
            return False

    def poll_joint_position(self, wait=False):
        """ Get the current joint positions of the hand.

        :param wait: If true, waits for a 'fresh' state reading.
        :return: Joint positions, or None if none have been received.
        """
        if wait:  # Clear joint state and wait for the next reading.
            self._joint_state = None
            while not self._joint_state:
                rospy.sleep(0.001)

        if self._joint_state:
            return (self._joint_state.position, self._joint_state.effort)
        else:
            return None

    def command_hand_configuration(self, hand_config):
        """
        Command a named hand configuration (e.g., pinch_index, envelop,
        gravity_compensation).

        The internal hand configuration names are defined in the
        AllegroNodeGrasp controller file. More human-friendly names are used
        by defining them as 'shortcuts' in the _named_grasps_mapping variable.
        Multiple strings can map to the same commanded configuration.

        :param hand_config: A human-friendly string of the desired
        configuration.
        :return: True if the grasp was known and commanded, false otherwise.
        """

        # Only use known named grasps.
        if hand_config in self._named_grasps_mappings:
            # Look up conversion of string -> msg
            msg = String(self._named_grasps_mappings[hand_config])
            rospy.logdebug('Commanding grasp: {}'.format(msg.data))
            self.pub_grasp.publish(msg)
            return True
        else:
            rospy.logwarn('Unable to command unknown grasp {}'.format(
                hand_config))
            return False

    def list_hand_configurations(self):
        """
        :return: List of valid strings for named hand configurations (including
        duplicates).
        """
        return self._named_grasps_mappings.keys()

    def set_envelop_torque(self, torque):
        """
        Command a specific envelop grasping torque.

        This only applies for the envelop named hand command. You can set the
        envelop torque before or after commanding the envelop grasp.

        :param torque: Desired torque, between 0 and 1. Values outside this
        range are clamped.
        :return: True.
        """

        torque = max(0.0, min(1.0, torque))  # Clamp within [0, 1]
        msg = Float32(torque)
        self.pub_envelop_torque.publish(msg)
        return True
