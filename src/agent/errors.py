"""Failures of a policy decision, distinct from transport and service failures."""


class PolicyExecutionError(Exception):
    """Consumes an evaluation episode; it must never trigger resampling."""


class PolicyValidationError(ValueError, PolicyExecutionError):
    """The policy supplied an invalid plan, argument or action."""


class PolicyProtocolError(RuntimeError, PolicyExecutionError):
    """The policy violated the tool/session execution protocol."""


def validate_policy_output(validator, *args, **kwargs):
    """Classify only validation of policy-supplied values, not arbitrary errors."""
    try:
        return validator(*args, **kwargs)
    except (ValueError, TypeError, KeyError, OverflowError) as error:
        raise PolicyValidationError(str(error)) from error
